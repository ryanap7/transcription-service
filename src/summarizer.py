"""
AI-powered summarization using Claude
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class AISummarizer:
    """Generate summaries using Claude AI"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize summarizer
        
        Args:
            api_key: Anthropic API key (optional)
            model: Claude model to use
        """
        self.api_key = api_key
        self.model = model
        self.client = None
        
        if api_key and api_key.strip() and not api_key.startswith("your_"):
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
                logger.info("✅ Claude summarizer initialized")
            except ImportError:
                logger.warning("Anthropic SDK not installed, summarization disabled")
            except Exception as e:
                logger.warning(f"Failed to initialize Claude: {e}")
        else:
            logger.info("Claude API key not provided, summarization disabled")
    
    def create_summary(self, transcript: str, stats: Dict) -> Optional[str]:
        """
        Generate AI summary of transcript
        
        Args:
            transcript: Full transcript text
            stats: Transcript statistics
            
        Returns:
            Summary text or None if summarization unavailable
        """
        if not self.client:
            logger.info("Summarization skipped (no API key)")
            return None
        
        try:
            logger.info("Generating summary with Claude...")
            
            # Truncate transcript if too long
            max_chars = 8000
            if len(transcript) > max_chars:
                transcript = transcript[:max_chars] + "\n\n[transcript truncated]"
                logger.info(f"Transcript truncated to {max_chars} chars")
            
            # Create prompt
            prompt = self._build_prompt(transcript, stats)
            
            # Call Claude API
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            summary = message.content[0].text
            logger.info("✅ Summary generated")
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return None
    
    @staticmethod
    def _build_prompt(transcript: str, stats: Dict) -> str:
        """
        Build prompt for Claude
        
        Args:
            transcript: Transcript text
            stats: Statistics dictionary
            
        Returns:
            Formatted prompt
        """
        # Format speaker stats
        speaker_info = []
        for speaker, data in sorted(stats.get('speakers', {}).items()):
            duration_min = data['duration'] / 60
            speaker_info.append(
                f"- {speaker}: {duration_min:.1f} menit "
                f"({data['percentage']:.1f}%), {data['words']} kata"
            )
        
        speakers_text = '\n'.join(speaker_info)
        
        prompt = f"""Analisis transkrip percakapan berikut dan berikan ringkasan yang komprehensif.

STATISTIK PERCAKAPAN:
- Durasi: {stats.get('total_duration', 0)/60:.1f} menit
- Jumlah pembicara: {stats.get('num_speakers', 0)}
- Total kata: {stats.get('total_words', 0):,}

KONTRIBUSI PEMBICARA:
{speakers_text}

TRANSKRIP:
{transcript}

Berikan analisis dalam format berikut:

## Ringkasan Utama
[2-3 kalimat ringkasan inti percakapan]

## Topik yang Dibahas
[3-5 poin topik utama yang dibahas, gunakan bullet points]

## Kontribusi Pembicara
[Analisis singkat peran dan kontribusi masing-masing pembicara]

## Kesimpulan dan Action Items
[Poin-poin kesimpulan penting dan action items jika ada]

Gunakan Bahasa Indonesia yang natural dan profesional."""
        
        return prompt