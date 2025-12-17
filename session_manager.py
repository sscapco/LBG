from typing import Dict, Optional
from data_models import GovernanceSessionState
import json
import os


class SessionManager:
    """
    Manages user sessions and their states
    """
    
    def __init__(self, cache_path: str = "session_cache.json"):
        self.cache_path = cache_path
        self.sessions: Dict[str, GovernanceSessionState] = {}
        self.load_sessions()
    
    def load_sessions(self):
        """Load sessions from cache file"""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    raw_sessions = json.load(f)
                
                # Convert to GovernanceSessionState objects
                for session_id, state_dict in raw_sessions.items():
                    self.sessions[session_id] = GovernanceSessionState(**state_dict)
                
                print(f"✅ Loaded {len(self.sessions)} sessions from cache")
            except Exception as e:
                print(f"⚠️ Failed to load session cache: {e}")
                self.sessions = {}
        else:
            print("ℹ️ No session cache found. Starting fresh.")
    
    def save_sessions(self):
        """Save sessions to cache file"""
        try:
            # Convert to dict format
            raw_sessions = {
                session_id: state.model_dump()
                for session_id, state in self.sessions.items()
            }
            
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(raw_sessions, f, indent=2)
            
            print(f"✅ Saved {len(self.sessions)} sessions to cache")
        except Exception as e:
            print(f"⚠️ Failed to save session cache: {e}")
    
    def get_session(self, session_id: str) -> Optional[GovernanceSessionState]:
        """Get session state for a session ID"""
        return self.sessions.get(session_id)
    
    def update_session(self, session_id: str, state: GovernanceSessionState):
        """Update session state"""
        self.sessions[session_id] = state
        self.save_sessions()
    
    def clear_session(self, session_id: str):
        """Clear a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.save_sessions()
            print(f"✅ Cleared session: {session_id}")
    
    def clear_all_sessions(self):
        """Clear all sessions"""
        self.sessions = {}
        self.save_sessions()
        print("✅ Cleared all sessions")
    
    def list_sessions(self) -> Dict[str, Dict]:
        """List all active sessions with their info"""
        return {
            session_id: {
                "last_focus_step": state.last_focus_step_id,
                "last_intent": state.last_intent,
                "timestamp": state.timestamp,
                "completed_steps": len(state.last_completed_ids)
            }
            for session_id, state in self.sessions.items()
        }


# Global session manager instance
session_manager = SessionManager()