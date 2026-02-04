"""
Transcript formatting and statistics
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class TranscriptFormatter:
    """Format and analyze transcripts"""
    
    @staticmethod
    def merge_consecutive(
        segments: List[Dict],
        gap_threshold: float = 1.0
    ) -> List[Dict]:
        """
        Merge consecutive segments from same speaker
        
        Args:
            segments: List of transcript segments
            gap_threshold: Maximum gap in seconds to merge (default: 1.0)
            
        Returns:
            Merged segments
        """
        if not segments:
            return []
        
        logger.info(f"Merging consecutive segments (gap_threshold={gap_threshold}s)...")
        
        merged = []
        current = segments[0].copy()
        
        for seg in segments[1:]:
            gap = seg['start'] - current['end']
            
            # Merge if same speaker and gap is small
            if seg['speaker'] == current['speaker'] and gap < gap_threshold:
                current['end'] = seg['end']
                current['duration'] = current['end'] - current['start']
                current['text'] += ' ' + seg['text']
            else:
                merged.append(current)
                current = seg.copy()
        
        # Add last segment
        merged.append(current)
        
        logger.info(f"Merged {len(segments)} -> {len(merged)} segments")
        return merged
    
    @staticmethod
    def format_text(segments: List[Dict], include_time: bool = True) -> str:
        """
        Format segments into readable transcript
        
        Args:
            segments: List of transcript segments
            include_time: Include timestamps in output
            
        Returns:
            Formatted transcript string
        """
        if not segments:
            return ""
        
        lines = []
        
        for seg in segments:
            if include_time:
                # Format time as MM:SS
                start_min = int(seg['start'] // 60)
                start_sec = int(seg['start'] % 60)
                end_min = int(seg['end'] // 60)
                end_sec = int(seg['end'] % 60)
                
                time_str = f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}]"
                lines.append(f"\n{seg['speaker']} {time_str}:")
            else:
                lines.append(f"\n{seg['speaker']}:")
            
            lines.append(seg['text'])
        
        return '\n'.join(lines)
    
    @staticmethod
    def get_stats(segments: List[Dict]) -> Dict:
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
        
        speakers = {}
        total_duration = 0
        total_words = 0
        
        for seg in segments:
            speaker = seg['speaker']
            duration = seg['duration']
            words = len(seg['text'].split())
            
            # Initialize speaker stats if not exists
            if speaker not in speakers:
                speakers[speaker] = {
                    'duration': 0,
                    'turns': 0,
                    'words': 0
                }
            
            # Update speaker stats
            speakers[speaker]['duration'] += duration
            speakers[speaker]['turns'] += 1
            speakers[speaker]['words'] += words
            
            # Update totals
            total_duration += duration
            total_words += words
        
        # Calculate percentages
        for speaker in speakers:
            speakers[speaker]['percentage'] = (
                speakers[speaker]['duration'] / total_duration * 100
                if total_duration > 0 else 0
            )
        
        return {
            'total_duration': total_duration,
            'total_words': total_words,
            'num_speakers': len(speakers),
            'speakers': speakers
        }