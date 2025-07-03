from typing import List, Dict, Optional
from datetime import datetime

class SimpleConversationMemory:
    """Ultra-simple conversation memory for recent Q&A pairs per session."""
    
    def __init__(self, max_history: int = 4):
        self.sessions = {}  
        self.max_history = max_history
    
    def add_qa_pair(self, question: str, answer: str, session_id: str = "default"):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        # Store question and truncated answer (to limit memory size)
        self.sessions[session_id].append({
            'q': question,
            'a': answer[:300] + "..." if len(answer) > 300 else answer,
            'time': datetime.now()
        })
        
        # Keep only last max_history pairs
        if len(self.sessions[session_id]) > self.max_history:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history:]
    
    def get_recent_context(self, session_id: str = "default", max_pairs: int = 2) -> str:
        history = self.sessions.get(session_id, [])
        if not history:
            return ""
        
        recent = history[-max_pairs:]
        context_parts = []
        for i, qa in enumerate(recent, 1):
            context_parts.append(f"Recent Q{i}: {qa['q']}")
            context_parts.append(f"Recent A{i}: {qa['a']}")
        return "\n".join(context_parts)
    
    def is_likely_followup(self, question: str) -> bool:
        q = question.lower().strip()
        words = q.split()
        if len(words) <= 2:
            return True
        followup_patterns = [
            'tell me more', 'more about', 'what about', 'how about',
            'and what', 'but what', 'also'
        ]
        pronouns = ['it', 'that', 'this', 'they', 'them', 'those', 'these']
        
        for pattern in followup_patterns:
            if pattern in q:
                return True
        
        for pronoun in pronouns:
            if q.startswith(pronoun + ' ') or q.startswith('what ' + pronoun) or q.startswith('how ' + pronoun):
                return True
        
        return False
    
    def clear_session(self, session_id: str = "default"):
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def get_stats(self) -> Dict:
        total_pairs = sum(len(history) for history in self.sessions.values())
        return {
            'active_sessions': len(self.sessions),
            'total_qa_pairs': total_pairs,
            'max_history_per_session': self.max_history
        }
