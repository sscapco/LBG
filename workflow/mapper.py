"""
Step mapping: Map parsed actions to concrete governance steps.
"""
import re
from typing import Optional, Tuple, List, Dict, Set

from core.models import IntentParseResult, MappedSteps, StepCandidate, GovernanceSessionState, ParsedAction
from core.embeddings import EmbeddingManager
from data.loader import GovernanceDataLoader
from data.cache import EmbeddingCache
from config import WorkflowConfig


class StepAliasBuilder:
    """Builds aliases for governance steps"""
    
    def __init__(self, data_loader: GovernanceDataLoader):
        self.data_loader = data_loader
        self.aliases: Dict[str, List[str]] = {}
        self._build_aliases()
    
    def _build_aliases(self):
        """Build alias mappings for all steps"""
        for step_id in self.data_loader.get_all_step_ids():
            record = self.data_loader.get_step_record(step_id)
            if not record:
                continue
            
            aliases = set()
            
            # Add step ID itself
            if step_id:
                aliases.add(step_id.lower())
            
            # Add step name
            name = record.name.strip()
            if name:
                aliases.add(name.lower())
                
                # Extract contents inside parentheses
                for match in re.findall(r"\(([^)]+)\)", name):
                    alias = match.strip()
                    if alias:
                        aliases.add(alias.lower())
                
                # Create acronym from uppercase letters
                caps = "".join(ch for ch in name if ch.isupper())
                if len(caps) >= 3:
                    aliases.add(caps.lower())
            
            self.aliases[step_id] = sorted(aliases)
    
    def get_aliases(self, step_id: str) -> List[str]:
        """Get all aliases for a step"""
        return self.aliases.get(step_id, [])
    
    def get_all_aliases(self) -> Dict[str, List[str]]:
        """Get all step aliases"""
        return self.aliases.copy()


class TextToStepMatcher:
    """Matches natural language text to governance steps"""
    
    def __init__(
        self,
        data_loader: GovernanceDataLoader,
        embedding_manager: EmbeddingManager,
        embedding_cache: EmbeddingCache,
        alias_builder: StepAliasBuilder
    ):
        self.data_loader = data_loader
        self.embedding_manager = embedding_manager
        self.embedding_cache = embedding_cache
        self.alias_builder = alias_builder
    
    def match(self, text: str, threshold: float = None) -> Tuple[Optional[str], str, float]:
        """
        Match text to a step using multiple strategies.
        
        Returns:
            (step_id, method, score) tuple
            method can be: "id_match", "alias_match", "substring_match", "embedding", "none"
        """
        if not text:
            return None, "none", 0.0
        
        threshold = threshold or WorkflowConfig.EMBEDDING_THRESHOLD
        t = text.strip().lower()
        
        if not t:
            return None, "none", 0.0
        
        # Strategy 1: Direct S-identifier (e.g., S12, S1, S123)
        m = re.search(r"\b(s\d{1,3})\b", t)
        if m:
            sid = m.group(1).upper()
            if sid in self.data_loader.get_all_step_ids():
                return sid, "id_match", 1.0
        
        # Strategy 2: Alias match
        for sid, aliases in self.alias_builder.get_all_aliases().items():
            for alias in aliases:
                if not alias:
                    continue
                # Try word boundary match first, then substring
                if re.search(r"\b" + re.escape(alias) + r"\b", t) or alias in t:
                    return sid, "alias_match", 1.0
        
        # Strategy 3: Substring match on step names
        best_sid = None
        best_len = 0
        for step_id in self.data_loader.get_all_step_ids():
            record = self.data_loader.get_step_record(step_id)
            if not record:
                continue
            name = record.name.lower()
            if name and name in t:
                if len(name) > best_len:
                    best_len = len(name)
                    best_sid = step_id
        
        if best_sid:
            return best_sid, "substring_match", 1.0
        
        # Strategy 4: Embedding-based semantic match
        action_emb = self.embedding_manager.get_embedding(text)
        all_embeddings = self.embedding_cache.get_all_embeddings()
        
        best_step = None
        best_score = 0.0
        for sid, step_emb in all_embeddings.items():
            score = self.embedding_manager.cosine_similarity(action_emb, step_emb)
            if score > best_score:
                best_score = score
                best_step = sid
        
        if best_step and best_score >= threshold:
            return best_step, "embedding", best_score
        
        return None, "none", best_score


class StepMapper:
    """Maps parsed intents and actions to concrete governance steps"""
    
    def __init__(
        self,
        data_loader: GovernanceDataLoader,
        embedding_manager: EmbeddingManager,
        embedding_cache: EmbeddingCache,
    ):
        self.data_loader = data_loader
        self.embedding_manager = embedding_manager
        self.embedding_cache = embedding_cache
        self.alias_builder = StepAliasBuilder(data_loader)
        self.matcher = TextToStepMatcher(
            data_loader,
            embedding_manager,
            embedding_cache,
            self.alias_builder
        )
    
    def map(
        self,
        user_message: str,
        parsed: IntentParseResult,
        previous_state: Optional[GovernanceSessionState]
    ) -> MappedSteps:
        """
        Map parsed intent and actions to concrete steps.
        
        Returns:
            MappedSteps with identified step IDs and metadata
        """
        intent = parsed.intent
        actions = parsed.actions
        focus_idx = parsed.focus_action_index
        is_starting = parsed.is_starting_question
        
        # Handle empty actions with context from previous state
        if not actions and previous_state:
            actions = self._create_synthetic_actions_from_state(
                previous_state,
                intent
            )
            if actions:
                focus_idx = 0
        
        # Map each action to steps
        completed_ids: List[str] = []
        in_progress_ids: List[str] = []
        referenced_ids: List[str] = []
        focus_step_candidate_id: Optional[str] = None
        has_direct_mapping = False
        
        for idx, action in enumerate(actions):
            step_id, method, score = self.matcher.match(action.description)
            if not step_id:
                continue
            
            # Track if we had a strong match
            if method in ("id_match", "alias_match", "substring_match"):
                has_direct_mapping = True
            
            # Categorize by status
            if action.status == "completed":
                completed_ids.append(step_id)
            elif action.status == "in_progress":
                in_progress_ids.append(step_id)
            
            # Add to referenced
            if step_id not in referenced_ids:
                referenced_ids.append(step_id)
            
            # Track focus
            if focus_idx is not None and idx == focus_idx:
                focus_step_candidate_id = step_id
        
        # Remove duplicates while preserving order
        completed_ids = self._unique_list(completed_ids)
        in_progress_ids = self._unique_list(in_progress_ids)
        referenced_ids = self._unique_list(referenced_ids)
        
        # Filter to valid step IDs
        valid_steps = set(self.data_loader.get_all_step_ids())
        completed_ids = [s for s in completed_ids if s in valid_steps]
        in_progress_ids = [s for s in in_progress_ids if s in valid_steps]
        referenced_ids = [s for s in referenced_ids if s in valid_steps]
        
        # Get semantic candidates from full message
        semantic_candidates = self._get_semantic_candidates(user_message)
        
        # Use semantic match as soft reference if nothing else found
        if not referenced_ids and semantic_candidates:
            top = semantic_candidates[0]
            if top.score >= 0.30:
                referenced_ids.append(top.step_id)
                if focus_step_candidate_id is None:
                    focus_step_candidate_id = top.step_id
        
        # Fallback to previous state if still nothing
        if not referenced_ids and previous_state:
            last_focus = previous_state.last_focus_step_id or previous_state.last_anchor_step_id
            if last_focus in valid_steps:
                referenced_ids.append(last_focus)
                if intent == "what_next" and last_focus not in completed_ids:
                    completed_ids.append(last_focus)
                focus_step_candidate_id = last_focus
        
        # Validate focus candidate
        if focus_step_candidate_id not in valid_steps:
            focus_step_candidate_id = None
        
        return MappedSteps(
            intent=intent,
            is_starting_question=is_starting,
            completed_ids=completed_ids,
            in_progress_ids=in_progress_ids,
            referenced_ids=referenced_ids,
            focus_step_candidate_id=focus_step_candidate_id,
            semantic_candidates=semantic_candidates,
            has_direct_mapping=has_direct_mapping,
        )
    
    def _create_synthetic_actions_from_state(
        self,
        previous_state: GovernanceSessionState,
        intent: str
    ) -> List[ParsedAction]:
        """Create synthetic actions from previous state"""
        last_focus = previous_state.last_focus_step_id or previous_state.last_anchor_step_id
        if not last_focus or last_focus not in self.data_loader.get_all_step_ids():
            return []
        
        record = self.data_loader.get_step_record(last_focus)
        desc = f"previous step {last_focus}: {record.name}" if record else f"previous step {last_focus}"
        status = "completed" if intent == "what_next" else "mentioned_only"
        
        return [ParsedAction(description=desc, status=status)]
    
    def _get_semantic_candidates(self, user_message: str, top_k: int = None) -> List[StepCandidate]:
        """Get top semantically similar steps"""
        top_k = top_k or WorkflowConfig.SEMANTIC_TOP_K
        
        try:
            msg_emb = self.embedding_manager.get_embedding(user_message)
            all_embeddings = self.embedding_cache.get_all_embeddings()
            
            scores = []
            for sid, emb in all_embeddings.items():
                score = self.embedding_manager.cosine_similarity(msg_emb, emb)
                scores.append((sid, score))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            top_results = scores[:top_k]
            
            return [StepCandidate(step_id=sid, score=score) for sid, score in top_results]
        except Exception:
            return []
    
    @staticmethod
    def _unique_list(items: List[str]) -> List[str]:
        """Remove duplicates while preserving order"""
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
