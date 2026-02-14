"""
Transcript formatting utilities
"""
from typing import List, Dict
from datetime import datetime


class TranscriptFormatter:
    """Format transcripts for output"""

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """
        Format seconds to HH:MM:SS

        Args:
            seconds: Time in seconds

        Returns:
            Formatted timestamp
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

    @staticmethod
    def format_segments_to_text(
        segments: List[Dict],
        include_timestamps: bool = True,
        include_speaker: bool = True
    ) -> str:
        """
        Format segments to readable text

        Args:
            segments: List of transcript segments
            include_timestamps: Whether to include timestamps
            include_speaker: Whether to include speaker labels

        Returns:
            Formatted transcript text
        """
        lines = []

        for seg in segments:
            parts = []

            # Add timestamp
            if include_timestamps:
                start = TranscriptFormatter.format_timestamp(seg['start'])
                end = TranscriptFormatter.format_timestamp(seg['end'])
                parts.append(f"[{start} - {end}]")

            # Add speaker
            if include_speaker:
                parts.append(f"{seg['speaker']}:")

            # Add text
            parts.append(seg['text'])

            lines.append(' '.join(parts))

        return '\n'.join(lines)

    @staticmethod
    def format_segments_to_srt(segments: List[Dict]) -> str:
        """
        Format segments to SRT subtitle format

        Args:
            segments: List of transcript segments

        Returns:
            SRT formatted text
        """
        srt_lines = []

        for i, seg in enumerate(segments, 1):
            # Sequence number
            srt_lines.append(str(i))

            # Timestamps in SRT format
            start = TranscriptFormatter._format_srt_timestamp(seg['start'])
            end = TranscriptFormatter._format_srt_timestamp(seg['end'])
            srt_lines.append(f"{start} --> {end}")

            # Text with speaker
            srt_lines.append(f"{seg['speaker']}: {seg['text']}")

            # Empty line
            srt_lines.append("")

        return '\n'.join(srt_lines)

    @staticmethod
    def _format_srt_timestamp(seconds: float) -> str:
        """
        Format timestamp for SRT format (HH:MM:SS,mmm)

        Args:
            seconds: Time in seconds

        Returns:
            SRT formatted timestamp
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def format_segments_to_vtt(segments: List[Dict]) -> str:
        """
        Format segments to WebVTT subtitle format

        Args:
            segments: List of transcript segments

        Returns:
            WebVTT formatted text
        """
        vtt_lines = ["WEBVTT", ""]

        for i, seg in enumerate(segments, 1):
            # Timestamps in WebVTT format
            start = TranscriptFormatter._format_vtt_timestamp(seg['start'])
            end = TranscriptFormatter._format_vtt_timestamp(seg['end'])
            vtt_lines.append(f"{start} --> {end}")

            # Text with speaker
            vtt_lines.append(f"{seg['speaker']}: {seg['text']}")

            # Empty line
            vtt_lines.append("")

        return '\n'.join(vtt_lines)

    @staticmethod
    def _format_vtt_timestamp(seconds: float) -> str:
        """
        Format timestamp for WebVTT format (HH:MM:SS.mmm)

        Args:
            seconds: Time in seconds

        Returns:
            WebVTT formatted timestamp
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    @staticmethod
    def calculate_statistics(segments: List[Dict]) -> Dict[str, any]:
        """
        Calculate transcript statistics

        Args:
            segments: List of transcript segments

        Returns:
            Dictionary with statistics
        """
        if not segments:
            return {
                'total_duration': 0,
                'total_words': 0,
                'num_speakers': 0,
                'speakers': {}
            }

        total_duration = segments[-1]['end'] - segments[0]['start']
        total_words = sum(len(seg['text'].split()) for seg in segments)

        # Speaker statistics
        speakers = {}
        for seg in segments:
            speaker = seg['speaker']

            if speaker not in speakers:
                speakers[speaker] = {
                    'duration': 0,
                    'words': 0,
                    'turns': 0,
                    'segments': []
                }

            speakers[speaker]['duration'] += seg['duration']
            speakers[speaker]['words'] += len(seg['text'].split())
            speakers[speaker]['turns'] += 1
            speakers[speaker]['segments'].append(seg)

        return {
            'total_duration': total_duration,
            'total_words': total_words,
            'num_speakers': len(speakers),
            'speakers': speakers
        }

    @staticmethod
    def create_formatted_report(
        segments: List[Dict],
        statistics: Dict[str, any],
        summary: str = None,
        audio_info: Dict[str, any] = None
    ) -> str:
        """
        Create a comprehensive formatted report

        Args:
            segments: Transcript segments
            statistics: Transcript statistics
            summary: AI-generated summary
            audio_info: Audio file information

        Returns:
            Formatted report text
        """
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("AUDIO TRANSCRIPTION REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Audio Information
        if audio_info:
            lines.append("AUDIO INFORMATION")
            lines.append("-" * 80)
            lines.append(f"Duration: {audio_info['duration_minutes']:.2f} minutes")
            lines.append(f"Sample Rate: {audio_info['sample_rate']} Hz")
            lines.append(f"Channels: {audio_info['channels']}")
            lines.append("")

        # Statistics
        lines.append("STATISTICS")
        lines.append("-" * 80)
        lines.append(f"Duration: {statistics['total_duration']/60:.2f} minutes")
        lines.append(f"Total Words: {statistics['total_words']:,}")
        lines.append(f"Number of Speakers: {statistics['num_speakers']}")
        lines.append("")

        # Speaker Breakdown
        lines.append("SPEAKER BREAKDOWN")
        lines.append("-" * 80)
        for speaker, data in sorted(statistics['speakers'].items()):
            duration_pct = (data['duration'] / statistics['total_duration']) * 100
            lines.append(f"{speaker}:")
            lines.append(f"  Talk Time: {data['duration']/60:.2f} min ({duration_pct:.1f}%)")
            lines.append(f"  Words: {data['words']:,}")
            lines.append(f"  Speaking Turns: {data['turns']}")
            lines.append("")

        # Summary
        if summary:
            lines.append("=" * 80)
            lines.append("AI SUMMARY")
            lines.append("=" * 80)
            lines.append(summary)
            lines.append("")

        # Full Transcript
        lines.append("=" * 80)
        lines.append("FULL TRANSCRIPT")
        lines.append("=" * 80)
        lines.append(TranscriptFormatter.format_segments_to_text(segments))
        lines.append("")

        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)

        return '\n'.join(lines)