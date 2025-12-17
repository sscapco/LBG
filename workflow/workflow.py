"""
Workflow computation: Determine next steps and workflow state.
"""
from typing import Optional, List, Dict
from core.models import MappedSteps, WorkflowState, StepCandidate
from data.loader import GovernanceDataLoader
from config import WorkflowConfig


class WorkflowComputer:
    """Computes workflow state and determines next steps"""
    
    def __init__(self, data_loader: GovernanceDataLoader):
        self.data_loader = data_loader
    
    def compute(self, mapped: MappedSteps) -> WorkflowState:
        """
        Compute complete workflow state from mapped steps.
        
        Returns:
            WorkflowState with focus, anchor, next steps, and prerequisites
        """
        intent = mapped.intent
        is_starting = mapped.is_starting_question
        completed_ids = mapped.completed_ids
        in_progress_ids = mapped.in_progress_ids
        referenced_ids = mapped.referenced_ids
        focus_candidate_id = mapped.focus_step_candidate_id
        semantic_candidates = mapped.semantic_candidates
        has_direct_mapping = mapped.has_direct_mapping
        
        # Find most advanced step in each category
        highest_completed = self._most_advanced(completed_ids)
        highest_in_progress = self._most_advanced(in_progress_ids)
        highest_referenced = self._most_advanced(referenced_ids)
        
        # Determine focus and anchor based on intent
        focus_step_id, anchor_step_id = self._determine_focus_and_anchor(
            intent,
            focus_candidate_id,
            highest_completed,
            highest_in_progress,
            highest_referenced
        )
        
        # Check for ambiguity
        needs_disambiguation = self._check_ambiguity(
            semantic_candidates,
            intent,
            has_direct_mapping,
            completed_ids,
            in_progress_ids
        )
        
        # Determine next steps
        next_step_ids = self._determine_next_steps(
            intent,
            anchor_step_id,
            focus_step_id,
            highest_completed,
            highest_referenced,
            referenced_ids,
            completed_ids,
            is_starting,
            needs_disambiguation
        )
        
        # If ambiguous, clear focus/anchor/next
        if needs_disambiguation:
            focus_step_id = None
            anchor_step_id = None
            next_step_ids = []
        
        # Find missing prerequisites
        missing_prereq_ids, prereq_edges = self._find_missing_prerequisites(
            next_step_ids,
            focus_step_id,
            completed_ids,
            in_progress_ids
        )
        
        # Gather step details
        step_details = self._gather_step_details(
            completed_ids,
            in_progress_ids,
            referenced_ids,
            next_step_ids,
            missing_prereq_ids,
            focus_step_id,
            anchor_step_id,
            semantic_candidates
        )
        
        return WorkflowState(
            intent=intent,
            is_starting_question=is_starting,
            completed_ids=completed_ids,
            in_progress_ids=in_progress_ids,
            referenced_ids=referenced_ids,
            anchor_step_id=anchor_step_id,
            focus_step_id=focus_step_id,
            next_step_ids=next_step_ids,
            missing_prereq_ids=missing_prereq_ids,
            step_details=step_details,
            prereq_edges=prereq_edges,
            semantic_candidates=semantic_candidates,
            needs_disambiguation=needs_disambiguation,
        )
    
    def _most_advanced(self, step_ids: List[str]) -> Optional[str]:
        """Find most advanced step in canonical order"""
        if not step_ids:
            return None
        return max(step_ids, key=lambda s: self.data_loader.get_step_index(s))
    
    def _determine_focus_and_anchor(
        self,
        intent: str,
        focus_candidate: Optional[str],
        highest_completed: Optional[str],
        highest_in_progress: Optional[str],
        highest_referenced: Optional[str]
    ) -> tuple:
        """Determine focus and anchor steps based on intent"""
        
        if intent == "ask_about_step":
            focus = focus_candidate or highest_referenced or highest_in_progress or highest_completed
            anchor = focus or highest_completed or highest_in_progress or highest_referenced
        
        elif intent == "help_current":
            focus = focus_candidate or highest_in_progress or highest_referenced or highest_completed
            anchor = focus or highest_completed or highest_in_progress or highest_referenced
        
        elif intent == "what_next":
            anchor = highest_completed or highest_in_progress or highest_referenced
            focus = None
        
        elif intent == "what_missed":
            anchor = highest_completed or highest_in_progress or highest_referenced
            focus = anchor
        
        else:  # "other"
            focus = focus_candidate or highest_referenced or highest_in_progress or highest_completed
            anchor = focus or highest_completed or highest_in_progress or highest_referenced
        
        return focus, anchor
    
    def _check_ambiguity(
        self,
        semantic_candidates: List[StepCandidate],
        intent: str,
        has_direct_mapping: bool,
        completed_ids: List[str],
        in_progress_ids: List[str]
    ) -> bool:
        """Check if query is ambiguous"""
        
        if len(semantic_candidates) < 2:
            return False
        
        if intent not in ("ask_about_step", "help_current", "what_next"):
            return False
        
        if has_direct_mapping:
            return False
        
        c1, c2 = semantic_candidates[0], semantic_candidates[1]
        
        # Ambiguous if both strong and close
        if (
            c1.step_id != c2.step_id
            and c1.score >= WorkflowConfig.AMBIGUITY_MIN_SCORE
            and c2.score >= WorkflowConfig.AMBIGUITY_SECONDARY_MIN_SCORE
            and (c1.score - c2.score) <= WorkflowConfig.AMBIGUITY_SCORE_DIFF_MAX
        ):
            # Only if no explicit progress mentioned
            if not completed_ids and not in_progress_ids:
                return True
        
        return False
    
    def _determine_next_steps(
        self,
        intent: str,
        anchor_step_id: Optional[str],
        focus_step_id: Optional[str],
        highest_completed: Optional[str],
        highest_referenced: Optional[str],
        referenced_ids: List[str],
        completed_ids: List[str],
        is_starting: bool,
        needs_disambiguation: bool
    ) -> List[str]:
        """Determine next steps in workflow"""
        
        if intent != "what_next" or needs_disambiguation:
            return []
        
        next_steps = []
        
        # Strategy 1: Use graph edges from anchor
        if anchor_step_id:
            outgoing = self.data_loader.get_outgoing_edges(anchor_step_id, "mandatory")
            for _, edge in outgoing.iterrows():
                to_step = edge["to"]
                if to_step in self.data_loader.get_all_step_ids() and to_step not in completed_ids:
                    if to_step not in next_steps:
                        next_steps.append(to_step)
        
        # Strategy 2: Starting question with no anchor
        if not anchor_step_id and not next_steps and is_starting:
            if referenced_ids:
                candidate = referenced_ids[0]
            else:
                all_steps = self.data_loader.get_all_step_ids()
                candidate = all_steps[0] if all_steps else None
            
            if candidate:
                next_steps = [candidate]
        
        # Strategy 3: Canonical next after highest completed
        if not next_steps and highest_completed:
            idx = self.data_loader.get_step_index(highest_completed)
            all_steps = self.data_loader.get_all_step_ids()
            if idx < len(all_steps) - 1:
                candidate = all_steps[idx + 1]
                if candidate not in completed_ids:
                    next_steps.append(candidate)
        
        # Strategy 4: Referenced but no progress
        if not next_steps and not completed_ids and highest_referenced and not is_starting:
            pass  # focus_step_id handled elsewhere
        
        return next_steps
    
    def _find_missing_prerequisites(
        self,
        next_step_ids: List[str],
        focus_step_id: Optional[str],
        completed_ids: List[str],
        in_progress_ids: List[str]
    ) -> tuple:
        """Find missing mandatory prerequisites"""
        
        missing_prereq_ids = []
        prereq_edges = {}
        
        targets = set(next_step_ids)
        if focus_step_id:
            targets.add(focus_step_id)
        
        for target in targets:
            incoming = self.data_loader.get_incoming_edges(target, "mandatory")
            for _, edge in incoming.iterrows():
                from_step = edge["from"]
                if (
                    from_step in self.data_loader.get_all_step_ids()
                    and from_step not in completed_ids
                    and from_step not in in_progress_ids
                ):
                    if from_step not in missing_prereq_ids:
                        missing_prereq_ids.append(from_step)
                    
                    prereq_edges.setdefault(target, []).append({
                        "from": from_step,
                        "question": edge.get("Question_Probe"),
                        "guard_condition": edge.get("guard_condition"),
                    })
        
        return missing_prereq_ids, prereq_edges
    
    def _gather_step_details(self, *step_id_lists) -> Dict[str, Dict]:
        """Gather details for all relevant steps"""
        relevant_ids = set()
        for id_list in step_id_lists:
            if isinstance(id_list, list):
                relevant_ids.update(id_list)
            elif isinstance(id_list, str) and id_list:
                relevant_ids.add(id_list)
            elif hasattr(id_list, '__iter__'):
                for item in id_list:
                    if hasattr(item, 'step_id'):
                        relevant_ids.add(item.step_id)
        
        step_details = {}
        for step_id in relevant_ids:
            record = self.data_loader.get_step_record(step_id)
            if record:
                step_details[step_id] = record.model_dump()
        
        return step_details
