/**
 * Why the auto-edit cut a word, in words a person can read.
 *
 * The backend records a `reason` on every word it removes. Both the transcript
 * panel (per word) and the render summary (as counts) show them, so the labels
 * live here rather than in either component.
 */

export const REASON_LABELS: Record<string, string> = {
  filler_sound: 'filler sounds heard in the audio',
  filler: 'filler words',
  stutter: 'stutters',
  retake: 'words from abandoned attempts (kept the final take)',
  false_start: 'false starts',
  soft_filler: 'crutch words',
  low_confidence: 'unclear mumbles',
  llm_filler: 'confirmed by the language model',
  not_fluent: 'words that broke the flow of the take',
  not_grammatical: 'debris that left a sentence unfinished',
  chatter: 'recording chatter',
  abandoned: 'abandoned sentences',
  repeat: 'repeated lines',
  drop: 'lines the AI editor removed',
  review: 'removed in your review',
};

/** The same reasons phrased for a single word, as a tooltip. */
export const REASON_TOOLTIPS: Record<string, string> = {
  filler_sound: 'A filler sound found in the audio that the transcript never had',
  filler: 'A filler word',
  stutter: 'A stutter — the next word repeats this one',
  retake: 'Part of an attempt the speaker abandoned; a later take is kept',
  false_start: 'Looks like a false start',
  soft_filler: 'A crutch word',
  low_confidence: 'The transcript is unsure what was said here',
  llm_filler: 'Removed on the language model’s judgement',
  not_fluent: 'Removed so the take reads as one continuous sentence',
  not_grammatical: 'Debris left over from a fumble, cut to complete the sentence',
  chatter: 'Recording chatter: talking to the camera or crew',
  abandoned: 'A sentence the speaker abandoned and never picked up again',
  repeat: 'Said again elsewhere',
  drop: 'Removed by the AI editor',
  review: 'Removed in your review of the cut',
};

export function reasonTooltip(reason?: string | null): string {
  if (!reason) return 'Removed by the auto-edit';
  return REASON_TOOLTIPS[reason] || `Removed: ${reason}`;
}

export function reasonLabel(reason: string): string {
  return REASON_LABELS[reason] || reason;
}
