"""
🧹 Code Cleaner Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Uses LLM to distinguish between valid Python code and explanatory text.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import logging
from cerebro.agents.base_agent import BaseAgent
from cerebro.llm import AutoBackend

logger = logging.getLogger(__name__)


class CodeCleanerAgent(BaseAgent):
    """
    Agent that cleans LLM-generated code by identifying and removing prose.
    
    Uses the LLM itself to classify each section as "code" or "text".
    """
    
    def __init__(self, llm: AutoBackend):
        super().__init__(llm, "CodeCleanerAgent")
    
    def clean_generated_code(self, code: str) -> str:
        """
        Clean LLM-generated code by removing prose and keeping only Python.
        
        Args:
            code: Raw LLM output that may contain markdown, prose, etc.
            
        Returns:
            Clean Python code with only valid statements and comments
        """
        logger.info("🧹 CodeCleanerAgent cleaning generated code...")
        
        # First pass: Simple cleanup of obvious issues
        lines = code.split('\n')
        
        # Remove markdown fences and obvious prose
        cleaned_lines = []
        in_prose_section = False
        for line in lines:
            stripped = line.strip()
            
            # Detect start of prose section
            if stripped.startswith('### '):
                in_prose_section = True
                continue
            
            # Exit prose section when we hit actual code again
            if in_prose_section:
                if stripped.startswith(('def ', 'class ', 'import ', 'from ', '#', '@')) or (stripped and not stripped.startswith('-')):
                    in_prose_section = False
                else:
                    continue
            
            if stripped in ['```', '```python', '```py']:
                continue
            if '<|im_end|>' in line or '<|endoftext|>' in line:
                continue
            cleaned_lines.append(line)
        
        # Group into sections (separated by blank lines)
        sections = []
        current_section = []
        for line in cleaned_lines:
            if line.strip() == '':
                if current_section:
                    sections.append('\n'.join(current_section))
                    current_section = []
                sections.append('')  # Keep blank line
            else:
                current_section.append(line)
        if current_section:
            sections.append('\n'.join(current_section))
        
        # Use LLM to classify sections
        final_lines = []
        for section in sections:
            if section == '':
                final_lines.append('')
                continue
            
            # Ask LLM: is this Python code?
            if self._is_python_code(section):
                final_lines.append(section)
            else:
                logger.debug(f"Removed prose: {section[:50]}...")
        
        result = '\n'.join(final_lines)
        
        # Remove excessive blank lines
        while '\n\n\n' in result:
            result = result.replace('\n\n\n', '\n\n')
        
        logger.info(f"✓ Cleaned: {len(code.splitlines())} → {len(result.splitlines())} lines")
        return result.strip()
    
    def _is_python_code(self, text: str) -> bool:
        """
        Use LLM to determine if text is valid Python code or prose.
        
        Args:
            text: Text to classify
            
        Returns:
            True if it's Python code, False if it's prose/explanation
        """
        # Quick heuristics first (avoid LLM call for obvious cases)
        stripped = text.strip()
        
        # Definitely code
        if any(stripped.startswith(kw) for kw in [
            'def ', 'class ', 'import ', 'from ', '@',
            'if ', 'for ', 'while ', 'try:', 'except', 'with ',
            'return ', 'yield ', 'raise ', 'assert '
        ]):
            return True
        
        # Definitely code (has assignment or function call)
        if '=' in stripped or '(' in stripped:
            return True
        
        # Definitely code (comment)
        if stripped.startswith('#'):
            return True
        
        # Definitely prose (numbered list, full sentences, markdown headings)
        if any([
            stripped[0].isdigit() and '. ' in stripped[:4],
            stripped.startswith(('This ', 'The ', 'Example ', 'Note:', '### ', '- The ', '- Basic ')),
            len(stripped.split()) > 15 and '.' in stripped and '(' not in stripped,
            '###' in stripped  # Markdown heading
        ]):
            return False
        
        # Ambiguous - ask LLM
        prompt = f"""Is this valid Python code or explanatory text?

Text:
{text}

Answer ONLY with: CODE or TEXT"""
        
        try:
            response = self.llm.reason(prompt, stream=False).strip().upper()
            return 'CODE' in response
        except:
            # If LLM fails, be conservative - keep it
            return True

