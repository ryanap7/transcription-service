"""
AI-powered summarization using Anthropic Claude - Optimized for concise output
"""
from typing import Optional, Dict
import anthropic

from src.core.config import Config
from src.core.exceptions import SummarizationError


class AISummarizer:
    """Generate concise summaries using Anthropic Claude"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize summarizer

        Args:
            api_key: Anthropic API key (uses config if not provided)
        """
        self.api_key = api_key or Config.ANTHROPIC_API_KEY
        self.client = None

        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)

    def is_available(self) -> bool:
        """Check if summarizer is available"""
        return self.client is not None

    def create_summary(
        self,
        transcript: str,
        statistics: Dict[str, any],
        language: str = 'id'
    ) -> Optional[str]:
        """
        Create concise AI summary of transcript - OPTIMIZED

        Args:
            transcript: Full transcript text
            statistics: Statistics about the transcript
            language: Output language ('id' for Indonesian, 'en' for English)

        Returns:
            Concise summary text or None if unavailable

        Raises:
            SummarizationError: If summarization fails
        """
        if not self.is_available():
            print("AI summarization not available (no API key)")
            return None

        try:
            print("Generating AI summary...")

            # Prepare concise prompt based on language
            if language == 'id':
                prompt = self._create_indonesian_prompt(transcript, statistics)
            else:
                prompt = self._create_english_prompt(transcript, statistics)

            # Call Claude API with reduced max_tokens for faster response
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=800,  # Reduced from 2000 for shorter, faster summaries
                temperature=0,  # Deterministic for speed
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            summary = message.content[0].text
            print("Summary generated successfully")

            return summary

        except Exception as e:
            raise SummarizationError(f"Failed to generate summary: {str(e)}")

    @staticmethod
    def _create_indonesian_prompt(transcript: str, statistics: Dict) -> str:
        """Create concise Indonesian prompt - OPTIMIZED"""
        # Limit transcript length for faster processing
        max_chars = 8000
        if len(transcript) > max_chars:
            transcript = transcript[:max_chars] + "..."

        return f"""Buatlah ringkasan SINGKAT dan PADAT dari transkrip berikut dalam bahasa Indonesia.

STATISTIK:
- Durasi: {statistics['total_duration']/60:.1f} menit
- Pembicara: {statistics['num_speakers']}
- Total kata: {statistics['total_words']:,}

TRANSKRIP:
{transcript}

INSTRUKSI:
- Maksimal 5-7 kalimat
- Fokus pada poin utama saja
- Langsung to the point
- Gunakan bullet points jika perlu
- Hindari penjelasan berlebihan

Format:
- Topik utama (1 kalimat)
- Poin-poin penting (3-5 bullet points)
- Kesimpulan (1 kalimat jika ada)"""

    @staticmethod
    def _create_english_prompt(transcript: str, statistics: Dict) -> str:
        """Create concise English prompt - OPTIMIZED"""
        # Limit transcript length for faster processing
        max_chars = 8000
        if len(transcript) > max_chars:
            transcript = transcript[:max_chars] + "..."

        return f"""Create a SHORT and CONCISE summary of this transcript in English.

STATISTICS:
- Duration: {statistics['total_duration']/60:.1f} minutes
- Speakers: {statistics['num_speakers']}
- Total words: {statistics['total_words']:,}

TRANSCRIPT:
{transcript}

INSTRUCTIONS:
- Maximum 5-7 sentences
- Focus on main points only
- Straight to the point
- Use bullet points if needed
- Avoid over-explanation

Format:
- Main topic (1 sentence)
- Key points (3-5 bullet points)
- Conclusion (1 sentence if applicable)"""

    def create_meeting_summary(
        self,
        transcript: str,
        statistics: Dict[str, any],
        meeting_type: str = 'general',
        language: str = 'id'
    ) -> Optional[str]:
        """
        Create concise meeting-specific summary - OPTIMIZED

        Args:
            transcript: Full transcript
            statistics: Transcript statistics
            meeting_type: Type of meeting ('general', 'standup', 'interview', etc.)
            language: Output language

        Returns:
            Summary text or None if unavailable
        """
        if not self.is_available():
            return None

        try:
            # Limit transcript length
            max_chars = 8000
            if len(transcript) > max_chars:
                transcript = transcript[:max_chars] + "..."

            # Customize prompt based on meeting type
            prompts = {
                'standup': self._standup_prompt(transcript, statistics, language),
                'interview': self._interview_prompt(transcript, statistics, language),
                'general': self._general_meeting_prompt(transcript, statistics, language)
            }

            prompt = prompts.get(meeting_type, prompts['general'])

            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=600,  # Even shorter for specific meeting types
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )

            return message.content[0].text

        except Exception as e:
            raise SummarizationError(f"Failed to generate meeting summary: {str(e)}")

    @staticmethod
    def _standup_prompt(transcript: str, statistics: Dict, language: str) -> str:
        """Concise prompt for standup meeting"""
        if language == 'id':
            return f"""Ringkas standup meeting ini SINGKAT:

{transcript}

Format (maksimal 3-4 kalimat per pembicara):
- Yang dikerjakan kemarin
- Yang akan dikerjakan hari ini
- Blocker (jika ada)"""
        else:
            return f"""Summarize this standup meeting BRIEFLY:

{transcript}

Format (max 3-4 sentences per speaker):
- Done yesterday
- Will do today
- Blockers (if any)"""

    @staticmethod
    def _interview_prompt(transcript: str, statistics: Dict, language: str) -> str:
        """Concise prompt for interview"""
        if language == 'id':
            return f"""Ringkas interview ini SINGKAT (maksimal 6 kalimat):

{transcript}

Fokus pada:
- Pertanyaan utama
- Jawaban kunci
- Penilaian singkat"""
        else:
            return f"""Summarize this interview BRIEFLY (max 6 sentences):

{transcript}

Focus on:
- Main questions
- Key answers
- Brief assessment"""

    @staticmethod
    def _general_meeting_prompt(transcript: str, statistics: Dict, language: str) -> str:
        """Concise prompt for general meeting"""
        if language == 'id':
            return f"""Ringkas meeting ini SINGKAT (maksimal 5-6 kalimat):

{transcript}

Sertakan:
- Topik utama (1 kalimat)
- Keputusan penting (2-3 poin)
- Action items (jika ada)"""
        else:
            return f"""Summarize this meeting BRIEFLY (max 5-6 sentences):

{transcript}

Include:
- Main topic (1 sentence)
- Key decisions (2-3 points)
- Action items (if any)"""