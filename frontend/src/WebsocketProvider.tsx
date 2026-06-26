import * as React from "react";
import { AppStateContext } from "./AppStateProvider";
import websocketHandler from "./websocketHandler";
import { TimerContext } from "./TimerProvider";
import memoize from "./memoize";
import WasmContext from "./WasmContext";
import { isWasmAvailable } from "./detectWasm";
import { GameMessage } from "./gen-types";

import type { JSX } from "react";

interface Context {
  send: (value: any) => void;
  reconnectNow: () => void;
}

export const WebsocketContext = React.createContext<Context>({
  send: () => {},
  reconnectNow: () => {},
});

interface IProps {
  children: JSX.Element[] | JSX.Element;
}

interface IBlobToArrayBufferQueue {
  enqueue: (blob: Blob, handler: (arr: ArrayBuffer) => void) => void;
}

const getFileReader: () => IBlobToArrayBufferQueue = memoize(() => {
  const queue: Array<{ blob: Blob; handler: (arr: ArrayBuffer) => void }> = [];
  const fr = new FileReader();
  fr.onload = () => {
    const next = queue.shift();
    if (next !== undefined) {
      next.handler(fr.result as ArrayBuffer);
      if (queue.length > 0) {
        fr.readAsArrayBuffer(queue[0].blob);
      }
    }
  };
  return {
    enqueue: (blob: Blob, handler: (arr: ArrayBuffer) => void) => {
      queue.push({ blob, handler });
      if (
        queue.length > 0 &&
        (fr.readyState === FileReader.EMPTY ||
          fr.readyState === FileReader.DONE)
      ) {
        fr.readAsArrayBuffer(queue[0].blob);
      }
    },
  };
});

const getBlobArrayBuffer: () => IBlobToArrayBufferQueue = memoize(() => {
  const queue: Array<{ blob: Blob; handler: (arr: ArrayBuffer) => void }> = [];
  const inflight: number[] = [];
  const onload = (arr: ArrayBuffer): void => {
    const next = queue.shift();
    if (next !== undefined) {
      inflight.shift();
      next.handler(arr);
      if (queue.length > 0) {
        inflight.push(0);
        queue[0].blob.arrayBuffer().then(onload, (err) => console.log(err));
      }
    }
  };
  return {
    enqueue: (blob: Blob, handler: (arr: ArrayBuffer) => void) => {
      queue.push({ blob, handler });
      if (inflight.length === 0 && queue.length > 0) {
        inflight.push(0);
        blob.arrayBuffer().then(onload, (err) => console.log(err));
      }
    },
  };
});

const WebsocketProvider: React.FunctionComponent<
  React.PropsWithChildren<IProps>
> = (props: IProps) => {
  const { state, updateState } = React.useContext(AppStateContext);
  const { decodeWireFormat } = React.useContext(WasmContext);
  const { setTimeout, clearTimeout } = React.useContext(TimerContext);
  const [timer, setTimer] = React.useState<number | null>(null);
  const [websocket, setWebsocket] = React.useState<WebSocket | null>(null);

  // Auto-reconnect bookkeeping. On an unexpected drop while seated in a room we
  // recreate the socket (exponential backoff) and replay the JoinRoom handshake
  // to reclaim the same seat. A `Kicked` message or a deliberate leave suppress
  // reconnection.
  const wsRef = React.useRef<WebSocket | null>(null);
  const deliberateCloseRef = React.useRef<boolean>(false);
  const reconnectAttemptsRef = React.useRef<number>(0);
  const reconnectTimerRef = React.useRef<number | null>(null);
  const connectRef = React.useRef<(isReconnect: boolean) => void>(() => {});

  // Because state/updateState are passed in and change every time something
  // happens, we need to maintain a reference to these props to prevent stale
  // closures which may happen if state/updateState is changed between when an
  // event listener is registered and when it fires.
  // https://reactjs.org/docs/hooks-faq.html#why-am-i-seeing-stale-props-or-state-inside-my-function
  const stateRef = React.useRef(state);
  const updateStateRef = React.useRef(updateState);
  const timerRef = React.useRef(timer);
  const setTimerRef = React.useRef(setTimer);
  const setTimeoutRef = React.useRef(setTimeout);
  const clearTimeoutRef = React.useRef(clearTimeout);

  React.useEffect(() => {
    stateRef.current = state;
    updateStateRef.current = updateState;
  }, [state, updateState]);

  React.useEffect(() => {
    setTimeoutRef.current = setTimeout;
    clearTimeoutRef.current = clearTimeout;
  }, [setTimeout, clearTimeout]);

  React.useEffect(() => {
    timerRef.current = timer;
    setTimerRef.current = setTimer;
  }, [timer, setTimerRef]);

  React.useEffect(() => {
    const computeUri = (): string => {
      // Resolution order for the game WebSocket host:
      //   1. window._WEBSOCKET_HOST  — injected at runtime by the backend's
      //      /runtime.js (set from the WEBSOCKET_HOST env var). Present for the
      //      single-service deploy.
      //   2. process.env.WEBSOCKET_HOST — baked into the bundle at build time via
      //      webpack DefinePlugin. Used by a standalone frontend (e.g. Vercel).
      //   3. same-origin — the original default.
      const runtimeWebsocketHost = (window as any)._WEBSOCKET_HOST;
      const bakedWebsocketHost = process.env.WEBSOCKET_HOST;
      return runtimeWebsocketHost !== undefined && runtimeWebsocketHost !== null
        ? runtimeWebsocketHost
        : bakedWebsocketHost !== undefined &&
            bakedWebsocketHost !== null &&
            bakedWebsocketHost !== ""
          ? bakedWebsocketHost
          : (location.protocol === "https:" ? "wss://" : "ws://") +
            location.host +
            location.pathname +
            (location.pathname.endsWith("/") ? "api" : "/api");
    };

    const scheduleReconnect = (): void => {
      // Only auto-reconnect after an UNEXPECTED drop while seated in a room.
      if (deliberateCloseRef.current) {
        return;
      }
      if (stateRef.current.roomName.length !== 16) {
        return;
      }
      if (reconnectTimerRef.current !== null) {
        return;
      }
      // A newer socket is already open/connecting (e.g. a stale socket closed
      // after we already reconnected) — don't pile on.
      const cur = wsRef.current;
      if (
        cur !== null &&
        (cur.readyState === WebSocket.OPEN ||
          cur.readyState === WebSocket.CONNECTING)
      ) {
        return;
      }
      updateStateRef.current({ reconnecting: true });
      const attempt = reconnectAttemptsRef.current;
      const delay = Math.min(1000 * 2 ** attempt, 10000);
      reconnectAttemptsRef.current = attempt + 1;
      reconnectTimerRef.current = window.setTimeout(() => {
        reconnectTimerRef.current = null;
        connectRef.current(true);
      }, delay);
    };

    const connect = (isReconnect: boolean): void => {
      const ws = new WebSocket(computeUri());
      wsRef.current = ws;
      setWebsocket(ws);

      ws.addEventListener("open", () => {
        reconnectAttemptsRef.current = 0;
        updateStateRef.current({
          connected: true,
          everConnected: true,
          reconnecting: false,
        });
        // On a reconnect the JoinRoom form (which normally sends the handshake)
        // isn't mounted, so replay it here to reclaim the seat by name.
        const { roomName, name } = stateRef.current;
        if (isReconnect && roomName.length === 16 && name.length > 0) {
          ws.send(
            JSON.stringify({
              room_name: roomName,
              name,
              disable_compression: !isWasmAvailable(),
            }),
          );
        }
      });

      ws.addEventListener("close", () => {
        updateStateRef.current({ connected: false });
        scheduleReconnect();
      });

      ws.addEventListener("message", (event: MessageEvent) => {
        if (timerRef.current !== null) {
          clearTimeoutRef.current(timerRef.current);
        }
        setTimerRef.current(null);

        // Check if the message is text (uncompressed JSON) or binary (compressed)
        if (typeof event.data === "string") {
          // Plain text JSON message (uncompressed)
          try {
            const message = JSON.parse(event.data);
            if ("Kicked" in message) {
              deliberateCloseRef.current = true;
              ws.close();
            } else {
              updateStateRef.current({
                connected: true,
                everConnected: true,
                reconnecting: false,
                ...websocketHandler(stateRef.current, message, (msg) => {
                  ws.send(JSON.stringify(msg));
                }),
              });
            }
          } catch (e) {
            console.error("Failed to parse JSON message:", e);
          }
        } else {
          // Binary message (compressed)
          const f = (buf: ArrayBuffer): void => {
            const message = decodeWireFormat(
              new Uint8Array(buf),
            ) as GameMessage;
            if (message && typeof message === "object" && "Kicked" in message) {
              deliberateCloseRef.current = true;
              ws.close();
            } else {
              updateStateRef.current({
                connected: true,
                everConnected: true,
                reconnecting: false,
                ...websocketHandler(stateRef.current, message, (msg) => {
                  ws.send(JSON.stringify(msg));
                }),
              });
            }
          };

          if (event.data.arrayBuffer !== undefined) {
            const b2a = getBlobArrayBuffer();
            b2a.enqueue(event.data, f);
          } else {
            const frs = getFileReader();
            frs.enqueue(event.data, f);
          }
        }
      });
    };

    connectRef.current = connect;
    connect(false);

    return () => {
      if (timerRef.current !== null) {
        clearTimeoutRef.current(timerRef.current);
      }
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
      }
    };
  }, []);

  const send = (value: any): void => {
    if (timerRef.current !== null) {
      clearTimeoutRef.current(timerRef.current);
    }
    // We expect a response back from the server within 5 seconds. Otherwise,
    // we should assume we have lost our websocket connection.

    const localTimerRef = setTimeoutRef.current(() => {
      if (timerRef.current === localTimerRef) {
        updateStateRef.current({ connected: false });
      }
    }, 5000);

    setTimerRef.current(localTimerRef);
    (wsRef.current ?? websocket)?.send(JSON.stringify(value));
  };

  // Manual "Reconnect now" — reset backoff and reconnect immediately, replaying
  // the JoinRoom handshake. Used by the disconnected/reconnecting UI.
  const reconnectNow = (): void => {
    deliberateCloseRef.current = false;
    reconnectAttemptsRef.current = 0;
    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    updateStateRef.current({ reconnecting: true });
    connectRef.current(true);
  };

  // TODO(read this from consumers instead of globals)
  (window as any).send = send;

  return (
    <WebsocketContext.Provider value={{ send, reconnectNow }}>
      {props.children}
    </WebsocketContext.Provider>
  );
};

export default WebsocketProvider;
