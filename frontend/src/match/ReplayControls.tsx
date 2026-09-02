/* Replay: move the match back to a given minute.

   The slider does not filter anything. It sets a `minute` parameter on the
   request, the server truncates the payload, and the browser only ever holds
   data that existed by that point. That is the whole design — a frontend that
   merely hides future events is one refresh away from showing them. */

import { useState } from 'react';
import { Badge, Segmented } from '../ui';

const STOPS = [0, 15, 30, 45, 60, 75, 90];

export function ReplayControls({
  minute,
  onChange,
  maxMinute,
  retrospective,
}: {
  minute: number | null;
  onChange: (minute: number | null) => void;
  maxMinute: number;
  retrospective: boolean;
}) {
  const [draft, setDraft] = useState(minute ?? maxMinute);
  // The momentum chart can move the replay position too, so the slider has to
  // follow a change it did not make. Adjusted during render rather than in an
  // effect: an effect here fires a second render for every scrub.
  const [seen, setSeen] = useState(minute);
  if (minute !== null && minute !== seen) {
    setSeen(minute);
    setDraft(minute);
  }

  const mode = minute === null ? 'full' : 'replay';

  return (
    <div className="replay-bar">
      <Segmented
        value={mode}
        onChange={(next) => onChange(next === 'full' ? null : draft)}
        options={[
          { value: 'full', label: 'Full match' },
          { value: 'replay', label: 'Replay' },
        ]}
      />

      {mode === 'replay' && (
        <>
          <input
            type="range"
            min={0}
            max={Math.max(maxMinute, 90)}
            step={1}
            value={draft}
            aria-label="Replay position in match minutes"
            onChange={(event) => setDraft(Number(event.target.value))}
            onMouseUp={() => onChange(draft)}
            onTouchEnd={() => onChange(draft)}
            onKeyUp={() => onChange(draft)}
            className="replay-slider"
          />
          <span className="replay-minute mono">{draft}′</span>
          <span className="row" style={{ gap: 4 }}>
            {STOPS.filter((stop) => stop <= Math.max(maxMinute, 90)).map((stop) => (
              <button
                key={stop}
                className={`ghost replay-stop ${draft === stop ? 'active' : ''}`}
                onClick={() => {
                  setDraft(stop);
                  onChange(stop);
                }}
              >
                {stop}′
              </button>
            ))}
          </span>
          {retrospective && (
            <Badge
              tone="warn"
              title="This match was ingested after full time, so a replay shows what happened by this minute — not what the platform knew at this minute."
            >
              retrospective
            </Badge>
          )}
        </>
      )}
    </div>
  );
}
