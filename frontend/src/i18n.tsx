import * as React from "react";

import type { JSX } from "react";

/*
 * Lightweight bilingual (中文 / English) i18n.
 *
 * We avoid a heavyweight dependency in this first pass and use a small React
 * context instead. Strings are referenced by key everywhere in the UI; full
 * coverage isn't required for M3 but the structure makes it easy to add.
 *
 * The chosen language persists in localStorage under "lang".
 */

export type Lang = "en" | "zh";

type Dict = Record<string, string>;

const en: Dict = {
  // App shell / generic
  "app.title.tractor": "Tractor",
  "app.title.findingFriends": "Finding Friends",
  "app.welcome":
    "Welcome! This website helps you play 升级 / Tractor / 找朋友 / Finding Friends with other people online.",
  "app.connecting": "Connecting to the server...",
  "app.disconnected":
    "It looks like you got disconnected from the server, please refresh! If the game is still ongoing, you should be able to re-join with the same name and pick up where you left off.",
  "app.refreshError":
    "An error has occurred, please try refreshing! If that doesn't resolve the issue, consider using the latest version of Mozilla Firefox or Google Chrome browsers.",
  "common.you": "You",
  "common.rank": "rank",

  // Theme / language / settings toggles
  "toolbar.theme.toLight": "Light theme",
  "toolbar.theme.toDark": "Dark theme",
  "toolbar.language": "中文",
  "toolbar.settings": "Settings",
  "toolbar.fourColor": "Four-color suits",

  // Join / lobby
  "join.roomName": "Room Name",
  "join.playerName": "Player Name",
  "join.enterRoom": "Enter a room code",
  "join.enterName": "Enter your name here",
  "join.generateRoom": "Generate new room",
  "join.join": "Join (or create) the game!",
  "join.intro":
    "Welcome to the game! Enter your name above to create a new game, or (re-)join the game if it already exists.",
  "join.readRules": "read the rules",
  "join.shareIntro":
    "Once you are in the game, share the room link with at least three friends to start playing!",
  "lobby.shareLink":
    "Send link to other players to allow them to join the game:",
  "lobby.start": "Start game",
  "lobby.waiting": "Waiting for players...",
  "lobby.randomize": "Randomize player order",

  // Add-AI control
  "ai.heading": "Add a computer player",
  "ai.add": "Add bot",
  "ai.remove": "Remove",
  "ai.difficulty": "Difficulty",
  "ai.difficulty.Easy": "Easy",
  "ai.difficulty.Medium": "Medium",
  "ai.difficulty.Hard": "Hard",
  "ai.difficulty.Omniscient": "Omniscient (sees all cards)",
  "ai.cheaterBadge": "CHEATER",
  "ai.cheaterWarning":
    "Omniscient bots cheat: they see every player's cards. For practice/testing only.",
  "ai.botLabel": "Bot",

  // Shengji terms
  "term.trump": "Trump",
  "term.level": "Level",
  "term.banker": "Declarer",
  "term.kitty": "Kitty",
  "term.tractor": "Tractor",
  "term.pair": "Pair",
  "term.throw": "Throw",
  "term.points": "Points",
  "term.noTrump": "No trump",

  // Play / status rail
  "rail.trump": "Trump",
  "rail.declarer": "Declarer",
  "rail.points": "Points",
  "rail.turn": "Turn",
  "rail.yourTurn": "Your turn",
  "rail.level": "Level",
  "play.finishTrick": "Finish trick",
  "play.takeBack": "Take back last play",
  "play.finishGame": "Finish game",
  "play.endEarly": "End game early",
  "play.yourHand": "Your hand",
  "play.confirmEndEarly":
    "Do you want to end the game early? There may still be points in the bottom...",
  "play.removedNote": "Note: these cards have been removed from the deck:",
  "play.cardsRemaining": "Cards remaining (that were not played):",
  "play.previousTrick": "Previous trick",
  "play.playSelected": "Play selected cards",
  "play.autoplay": "Autoplay selected cards",
  "play.cancelAutoplay": "Don't autoplay selected cards",
  "play.tapToConfirm": "Tap selected cards or press play to confirm",
  "play.multiInterpretation":
    "It looks like you are making a play that can be interpreted in multiple ways!",
  "play.whichDidYouMean": "Which of the following did you mean?",
  "play.seat.you": "You (bottom)",
  "play.seat.left": "Left",
  "play.seat.across": "Across",
  "play.seat.right": "Right",
  "play.waitingTurn": "Waiting for other players...",

  // Trump display
  "trump.suitIs": "The trump suit is",
  "trump.rank": "rank",
  "trump.noTrumpRank": "No trump, rank {rank}",
  "trump.noTrump": "No trump",

  // Bidding
  "bid.title": "Bidding",
  "bid.round": "Bids (round {round} of bidding)",
  "bid.remaining": "Bids ({count} cards remaining in the deck)",
  "bid.fromBottom": "(from bottom)",
  "bid.noBidsYet": "No bids yet...",
  "bid.noBidsInNoTrump": "No bidding in no trump!",
  "bid.clickToBid": "Click a bid option to bid",
  "bid.loading": "Loading bid options...",
  "bid.noAvailable": "No available bids!",
  "bid.option": "Bid option {n}",
  "bid.takeBack": "Take back bid",
  "bid.drawCard": "Draw card",
  "bid.autoDraw": "auto-draw",
  "bid.pickUpKitty": "Pick up cards from the bottom",
  "bid.revealCard": "Reveal card from the bottom",

  // Exchange
  "exchange.yourHand": "Your hand",
  "exchange.discarded": "Discarded cards {n} / {total}",
  "exchange.finalize": "Finalize exchanged cards",
  "exchange.start": "Start game",
  "exchange.pickFriends": "Pick friends",
  "exchange.waiting": "Waiting...",

  // Friends
  "friends.intro": "The person to play the {nth} {card} is a friend.",
  "friends.played": "{n} played in previous tricks.",

  // Points
  "points.title": "Points",
  "points.loading": "Loading scores...",
  "points.stolenFrom": "stolen from {name}'s team.",
  "points.nextThreshold": "(next threshold: {n}分)",
  "points.willGoUp": "{name}'s team will go up {n} level(s)",
  "points.smallTeamBonus": ", including a small-team bonus",
  "points.neitherUp": "Neither team will go up a level",
  "points.attackingUp": "The attacking team will go up {n} level(s)",
};

const zh: Dict = {
  "app.title.tractor": "升级",
  "app.title.findingFriends": "找朋友",
  "app.welcome":
    "欢迎！本网站可以让你和朋友们在线游玩 升级 / 拖拉机 / 找朋友。",
  "app.connecting": "正在连接服务器…",
  "app.disconnected":
    "你似乎与服务器断开了连接，请刷新页面！如果游戏仍在进行，你可以用相同的名字重新加入并继续。",
  "app.refreshError":
    "出现了错误，请尝试刷新页面！如果仍未解决，请使用最新版本的 Firefox 或 Chrome 浏览器。",
  "common.you": "你",
  "common.rank": "级",

  "toolbar.theme.toLight": "浅色主题",
  "toolbar.theme.toDark": "深色主题",
  "toolbar.language": "English",
  "toolbar.settings": "设置",
  "toolbar.fourColor": "四色花色",

  "join.roomName": "房间名",
  "join.playerName": "玩家名",
  "join.enterRoom": "输入房间号",
  "join.enterName": "在此输入你的名字",
  "join.generateRoom": "生成新房间",
  "join.join": "加入（或创建）游戏！",
  "join.intro":
    "欢迎！在上方输入你的名字来创建新游戏，或重新加入已存在的游戏。",
  "join.readRules": "阅读规则",
  "join.shareIntro": "进入游戏后，把房间链接分享给至少三位朋友即可开始！",
  "lobby.shareLink": "把链接发送给其他玩家，邀请他们加入游戏：",
  "lobby.start": "开始游戏",
  "lobby.waiting": "等待玩家中…",
  "lobby.randomize": "随机排序玩家",

  "ai.heading": "添加电脑玩家",
  "ai.add": "添加机器人",
  "ai.remove": "移除",
  "ai.difficulty": "难度",
  "ai.difficulty.Easy": "简单",
  "ai.difficulty.Medium": "中等",
  "ai.difficulty.Hard": "困难",
  "ai.difficulty.Omniscient": "全知（作弊）",
  "ai.cheaterBadge": "作弊",
  "ai.cheaterWarning":
    "全知机器人会作弊：它能看到所有玩家的牌。仅供练习/测试使用。",
  "ai.botLabel": "机器人",

  "term.trump": "主牌",
  "term.level": "级牌",
  "term.banker": "庄家",
  "term.kitty": "底牌",
  "term.tractor": "拖拉机",
  "term.pair": "对子",
  "term.throw": "甩牌",
  "term.points": "分",
  "term.noTrump": "无主",

  "rail.trump": "主牌",
  "rail.declarer": "庄家",
  "rail.points": "分",
  "rail.turn": "出牌",
  "rail.yourTurn": "轮到你了",
  "rail.level": "级牌",
  "play.finishTrick": "结束本轮",
  "play.takeBack": "收回上次出牌",
  "play.finishGame": "结束游戏",
  "play.endEarly": "提前结束游戏",
  "play.yourHand": "你的手牌",
  "play.confirmEndEarly": "你确定要提前结束游戏吗？底牌里可能还有分…",
  "play.removedNote": "注意：以下牌已从牌堆中移除：",
  "play.cardsRemaining": "剩余未出的牌：",
  "play.previousTrick": "上一轮",
  "play.playSelected": "出选中的牌",
  "play.autoplay": "自动出选中的牌",
  "play.cancelAutoplay": "取消自动出牌",
  "play.tapToConfirm": "再次点击选中的牌或按出牌确认",
  "play.multiInterpretation": "你的出牌可以有多种理解方式！",
  "play.whichDidYouMean": "你指的是以下哪一种？",
  "play.seat.you": "你（下家）",
  "play.seat.left": "左家",
  "play.seat.across": "对家",
  "play.seat.right": "右家",
  "play.waitingTurn": "等待其他玩家…",

  "trump.suitIs": "主牌花色为",
  "trump.rank": "级",
  "trump.noTrumpRank": "无主，级牌 {rank}",
  "trump.noTrump": "无主",

  "bid.title": "叫主",
  "bid.round": "叫牌（第 {round} 轮）",
  "bid.remaining": "叫牌（牌堆还剩 {count} 张）",
  "bid.fromBottom": "（来自底牌）",
  "bid.noBidsYet": "还没有人叫牌…",
  "bid.noBidsInNoTrump": "无主时不能叫牌！",
  "bid.clickToBid": "点击一个选项来叫牌",
  "bid.loading": "正在加载叫牌选项…",
  "bid.noAvailable": "没有可用的叫牌！",
  "bid.option": "叫牌选项 {n}",
  "bid.takeBack": "收回叫牌",
  "bid.drawCard": "摸牌",
  "bid.autoDraw": "自动摸牌",
  "bid.pickUpKitty": "拿起底牌",
  "bid.revealCard": "翻开底牌",

  "exchange.yourHand": "你的手牌",
  "exchange.discarded": "已埋的牌 {n} / {total}",
  "exchange.finalize": "确认埋牌",
  "exchange.start": "开始游戏",
  "exchange.pickFriends": "选择朋友",
  "exchange.waiting": "等待中…",

  "friends.intro": "出第 {nth} 张 {card} 的玩家是朋友。",
  "friends.played": "已在之前的回合中出了 {n} 张。",

  "points.title": "分数",
  "points.loading": "正在加载分数…",
  "points.stolenFrom": "从 {name} 的队伍中夺得。",
  "points.nextThreshold": "（下一档：{n}分）",
  "points.willGoUp": "{name} 的队伍将升 {n} 级",
  "points.smallTeamBonus": "，包含小队加成",
  "points.neitherUp": "两队都不升级",
  "points.attackingUp": "进攻方将升 {n} 级",
};

const dictionaries: Record<Lang, Dict> = { en, zh };

const STORAGE_KEY = "lang";

const loadLang = (): Lang => {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored === "en" || stored === "zh") {
      return stored;
    }
    // Fall back to the browser preference if it's Chinese.
    if (
      typeof navigator !== "undefined" &&
      navigator.language &&
      navigator.language.toLowerCase().startsWith("zh")
    ) {
      return "zh";
    }
  } catch {
    // ignore
  }
  return "en";
};

export type TranslateFn = (
  key: string,
  vars?: Record<string, string | number>,
) => string;

interface I18nContextValue {
  lang: Lang;
  setLang: (lang: Lang) => void;
  toggleLang: () => void;
  t: TranslateFn;
}

const noop = (): void => {};

export const I18nContext = React.createContext<I18nContextValue>({
  lang: "en",
  setLang: noop,
  toggleLang: noop,
  t: (key) => key,
});

interface IProps {
  children: React.ReactNode;
}

export const I18nProvider = (props: IProps): JSX.Element => {
  const [lang, setLangState] = React.useState<Lang>(() => loadLang());

  const setLang = React.useCallback((next: Lang) => {
    setLangState(next);
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // ignore
    }
  }, []);

  React.useEffect(() => {
    document.documentElement.setAttribute("lang", lang === "zh" ? "zh" : "en");
  }, [lang]);

  const value = React.useMemo<I18nContextValue>(() => {
    const t: TranslateFn = (key, vars) => {
      const dict = dictionaries[lang];
      let str = dict[key] ?? dictionaries.en[key] ?? key;
      if (vars) {
        Object.entries(vars).forEach(([k, v]) => {
          str = str.replace(new RegExp(`\\{${k}\\}`, "g"), String(v));
        });
      }
      return str;
    };
    return {
      lang,
      setLang,
      toggleLang: () => setLang(lang === "en" ? "zh" : "en"),
      t,
    };
  }, [lang, setLang]);

  return (
    <I18nContext.Provider value={value}>{props.children}</I18nContext.Provider>
  );
};

export const useTranslation = (): I18nContextValue =>
  React.useContext(I18nContext);

/*
 * Non-hook translate helper for legacy class components (Draw / Exchange) that
 * can't consume the React context. It reads the persisted language directly.
 * Prefer `useTranslation` in function components.
 */
export const translate: TranslateFn = (key, vars) => {
  const lang = loadLang();
  const dict = dictionaries[lang];
  let str = dict[key] ?? dictionaries.en[key] ?? key;
  if (vars) {
    Object.entries(vars).forEach(([k, v]) => {
      str = str.replace(new RegExp(`\\{${k}\\}`, "g"), String(v));
    });
  }
  return str;
};
