/** Shared studio / workspace UI bits (Pipeline Studio + settings). */

/** Sentinel `<option value>`s so async lists still render as real `<select>` controls. */
export const STUDIO_SELECT_LOADING = "__studio_loading__";
export const STUDIO_SELECT_EMPTY = "__studio_empty__";

export type StudioUpdatingBadgeProps = {
  active: boolean;
  /** Short status line shown next to the pulse dot */
  label?: string;
};

/**
 * Always stays in DOM (uses visibility:hidden when inactive) so flex/grid siblings
 * never resize or rewrap when loading state changes.
 */
export function StudioUpdatingBadge({ active, label = "Updating ..." }: StudioUpdatingBadgeProps) {
  return (
    <span
      className="studio-updating-badge"
      role="status"
      aria-live="polite"
      aria-hidden={active ? undefined : "true"}
      style={active ? undefined : { visibility: "hidden" }}
    >
      <span className="studio-updating-badge__dot" aria-hidden />
      {label}
    </span>
  );
}

export type SizeStableLabelProps = {
  label: string;
  busyLabel: string;
  isBusy: boolean;
};

/**
 * Keeps a button's rendered width stable while its label switches between two states.
 * Both texts sit in the same CSS grid cell so the button is always as wide as the
 * longer of the two strings; only the active text is visible.
 */
export function SizeStableLabel({ label, busyLabel, isBusy }: SizeStableLabelProps) {
  return (
    <span style={{ display: "inline-grid" }}>
      <span style={{ gridArea: "1/1", visibility: isBusy ? "hidden" : "visible" }}>{label}</span>
      <span style={{ gridArea: "1/1", visibility: isBusy ? "visible" : "hidden" }}>{busyLabel}</span>
    </span>
  );
}
