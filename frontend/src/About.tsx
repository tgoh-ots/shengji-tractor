import * as React from "react";
import ReactModal from "react-modal";
import { useTranslation } from "./i18n";

import type { JSX } from "react";

/*
 * About / 关于 modal.
 *
 * A small "how this was built" blurb plus a Sources list (game rules, the
 * original open-source engine this project is forked from, and the AI research
 * that informed the bots). All copy is bilingual (English / 中文) and is kept
 * SELF-CONTAINED in this file -- it is deliberately NOT routed through i18n.tsx
 * so the About content can evolve independently of the shared dictionary.
 *
 * Styling matches the rest of the M3 redesign: the ReactModal content surface
 * is themed globally (see .ReactModal__Content in style.css), and we reuse the
 * `sj-*` token classes (see theme.css) for the close button.
 */

const contentStyle: React.CSSProperties = {
  position: "absolute",
  top: "50%",
  left: "50%",
  width: "min(560px, 92vw)",
  transform: "translate(-50%, -50%)",
  padding: "1.5rem",
};

type Lang = "en" | "zh";

interface Copy {
  title: string;
  builtHeading: string;
  builtBody: string;
  sourcesHeading: string;
  sources: { label: string; href: string; note: string }[];
  licenseNote: string;
  close: string;
}

const COPY: Record<Lang, Copy> = {
  en: {
    title: "About / 关于",
    builtHeading: "How this was built",
    builtBody:
      'Shengji Online (升级 Online) is a free, open-source way to play 升级 / Tractor / Finding Friends with friends online. It is forked from the open-source rbtying/shengji engine (MIT). On top of that foundation it adds computer opponents — a heuristic + search AI backed by a distilled learned network, plus an optional perfect-information ("omniscient") mode — a fully redesigned, responsive bilingual interface, and deployment on Fly.io.',
    sourcesHeading: "Sources",
    sources: [
      {
        label: 'Wikipedia: "Sheng ji"',
        href: "https://en.wikipedia.org/wiki/Sheng_ji",
        note: "Game overview & rules",
      },
      {
        label: "pagat.com: Tractor (Tuo La Ji)",
        href: "https://www.pagat.com/kt5/tractor.html",
        note: "Detailed rules reference",
      },
      {
        label: "github.com/rbtying/shengji",
        href: "https://github.com/rbtying/shengji",
        note: "Original game engine (MIT) — the foundation of this fork",
      },
      {
        label: "Berkeley EECS-2023-127: ShengJi+",
        href: "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2023/EECS-2023-127.html",
        note: "AI research for trick-taking games",
      },
      {
        label: "DouZero",
        href: "https://github.com/kwai/DouZero",
        note: "Deep RL for related Chinese card games",
      },
    ],
    licenseNote:
      "Released under the MIT License. The original copyright is preserved — see LICENSE and NOTICE.",
    close: "Close",
  },
  zh: {
    title: "关于 / About",
    builtHeading: "本项目如何构建",
    builtBody:
      "升级 Online（Shengji Online）是一个免费、开源的在线平台，可与好友一起玩升级 / 拖拉机 / 找朋友。它基于开源的 rbtying/shengji 引擎（MIT 许可）二次开发。在此基础上，增加了电脑对手——结合启发式搜索与蒸馏后的神经网络，并提供可选的完全信息（“全知”）模式——全新设计的自适应中英双语界面，并部署在 Fly.io 上。",
    sourcesHeading: "资料来源",
    sources: [
      {
        label: "维基百科：《Sheng ji》",
        href: "https://en.wikipedia.org/wiki/Sheng_ji",
        note: "游戏概览与规则",
      },
      {
        label: "pagat.com：拖拉机（Tractor）",
        href: "https://www.pagat.com/kt5/tractor.html",
        note: "详细规则参考",
      },
      {
        label: "github.com/rbtying/shengji",
        href: "https://github.com/rbtying/shengji",
        note: "原始游戏引擎（MIT）——本 fork 的基础",
      },
      {
        label: "Berkeley EECS-2023-127：ShengJi+",
        href: "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2023/EECS-2023-127.html",
        note: "用于桌牌类游戏的 AI 研究",
      },
      {
        label: "DouZero",
        href: "https://github.com/kwai/DouZero",
        note: "相关中文纸牌游戏的深度强化学习",
      },
    ],
    licenseNote:
      "以 MIT 许可证发布。原始版权声明已保留——详见 LICENSE 与 NOTICE。",
    close: "关闭",
  },
};

interface IProps {
  isOpen: boolean;
  onRequestClose: () => void;
}

const About = (props: IProps): JSX.Element => {
  const { lang } = useTranslation();
  const copy = COPY[lang === "zh" ? "zh" : "en"];

  return (
    <ReactModal
      isOpen={props.isOpen}
      onRequestClose={props.onRequestClose}
      shouldCloseOnOverlayClick
      shouldCloseOnEsc
      style={{ content: contentStyle }}
      contentLabel={copy.title}
    >
      <div className="sj-about text-[var(--text-primary)]">
        <h2 className="m-0 text-xl font-bold tracking-tight">{copy.title}</h2>

        <h3 className="mb-1 mt-4 text-sm font-bold uppercase tracking-wide text-[var(--text-secondary)]">
          {copy.builtHeading}
        </h3>
        <p className="m-0 text-sm leading-relaxed text-[var(--text-primary)]">
          {copy.builtBody}
        </p>

        <h3 className="mb-1 mt-4 text-sm font-bold uppercase tracking-wide text-[var(--text-secondary)]">
          {copy.sourcesHeading}
        </h3>
        <ul className="m-0 list-disc space-y-1 pl-5 text-sm leading-relaxed">
          {copy.sources.map((s) => (
            <li key={s.href}>
              <a
                href={s.href}
                target="_blank"
                rel="noreferrer"
                className="text-[var(--accent)] underline underline-offset-2"
              >
                {s.label}
              </a>{" "}
              <span className="text-[var(--text-secondary)]">— {s.note}</span>
            </li>
          ))}
        </ul>

        <p className="mt-4 text-xs text-[var(--text-secondary)]">
          {copy.licenseNote}
        </p>

        <div className="mt-5 text-right">
          <button
            type="button"
            className="sj-btn sj-btn-primary"
            onClick={props.onRequestClose}
          >
            {copy.close}
          </button>
        </div>
      </div>
    </ReactModal>
  );
};

export default About;
