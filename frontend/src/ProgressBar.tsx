import * as React from "react";
import { useTranslation } from "./i18n";

import type { JSX } from "react";

interface IProps {
  checkpoints: number[];
  totalPoints: number;
  challengerPoints: number;
  landlordPoints: number;
  hideLandlordPoints: boolean;
  // Whether the local player sits on the landlord (declarer) side. Drives the
  // "Your team" vs "Opponents" labelling so the local player can tell at a
  // glance which color is theirs. Spectators leave this undefined → neutral
  // "Declarer's side" / "Opponents" labels (mirroring the in-game legend).
  selfOnLandlordTeam?: boolean;
}

const challengerColor = "#5bc0de";
const landlordColor = "#d9534f";

const clampPct = (proportion: number): number =>
  Math.max(0, Math.min(1, proportion)) * 100;

/*
 * Scoring progress bar (dark-mode friendly).
 *
 * A single horizontal track shows how many points the attacking ("challenger")
 * team has collected. Threshold markers (the level-up checkpoints) are laid out
 * as short ticks with their value labelled BELOW the track, so nothing overlaps
 * and the scale stays legible. When landlord points are visible, a second
 * marker shows their progress from the top end.
 */
const ProgressBar = (props: IProps): JSX.Element => {
  const { t } = useTranslation();
  const { totalPoints, challengerPoints, landlordPoints, checkpoints } = props;

  const challengerPct = clampPct(challengerPoints / totalPoints);
  // Landlord points count down from the top of the scale.
  const landlordMarker = totalPoints - landlordPoints;
  const landlordPct = clampPct(landlordMarker / totalPoints);

  // The challenger (blue) is the attacking side; the landlord (red) is the
  // declarer's side. Label each swatch so the local player tells at a glance
  // which color is theirs. Falls back to the neutral side names for spectators.
  const challengerLabel =
    props.selfOnLandlordTeam === undefined
      ? t("team.legend.opponents")
      : props.selfOnLandlordTeam
        ? t("points.opponents")
        : t("points.yourTeam");
  const landlordLabel =
    props.selfOnLandlordTeam === undefined
      ? t("team.legend.declarerSide")
      : props.selfOnLandlordTeam
        ? t("points.yourTeam")
        : t("points.opponents");

  return (
    <div className="sj-score-bar">
      <div className="sj-score-track" aria-hidden="true">
        <div
          className="sj-score-fill"
          style={{
            width: `${challengerPct}%`,
            backgroundColor: challengerColor,
          }}
        />
        {/* Threshold ticks sit on top of the track. */}
        {checkpoints.map((checkpoint, i) => {
          const reached = challengerPoints >= checkpoint;
          return (
            <span
              key={i}
              className="sj-score-tick"
              style={{
                left: `${clampPct(checkpoint / totalPoints)}%`,
                backgroundColor: reached ? challengerColor : undefined,
              }}
            />
          );
        })}
        {/* Current challenger position marker. */}
        <span
          className="sj-score-marker"
          style={{
            left: `${challengerPct}%`,
            borderColor: challengerColor,
          }}
        />
        {!props.hideLandlordPoints && (
          <span
            className="sj-score-marker"
            style={{ left: `${landlordPct}%`, borderColor: landlordColor }}
          />
        )}
      </div>

      {/* Numeric scale: 0 … thresholds … total, labelled below the track. */}
      <div className="sj-score-scale" aria-hidden="true">
        <span className="sj-score-scale-end" style={{ left: "0%" }}>
          0
        </span>
        {checkpoints.map((checkpoint, i) => (
          <span
            key={i}
            className="sj-score-scale-label"
            style={{ left: `${clampPct(checkpoint / totalPoints)}%` }}
          >
            {checkpoint}
          </span>
        ))}
        <span className="sj-score-scale-end" style={{ left: "100%" }}>
          {totalPoints}
        </span>
      </div>

      {/* Legend so the two colored markers are self-explanatory: each swatch is
       * labelled with the team it belongs to AND its point total. */}
      <div className="sj-score-legend">
        <span className="sj-score-legend-item">
          <span
            className="sj-score-swatch"
            style={{ backgroundColor: challengerColor }}
          />
          <span>
            {challengerLabel}: {challengerPoints}
            {t("term.fenUnit")}
          </span>
        </span>
        {!props.hideLandlordPoints && (
          <span className="sj-score-legend-item">
            <span
              className="sj-score-swatch"
              style={{ backgroundColor: landlordColor }}
            />
            <span>
              {landlordLabel}: {landlordMarker}
              {t("term.fenUnit")}
            </span>
          </span>
        )}
      </div>
    </div>
  );
};

export default ProgressBar;
