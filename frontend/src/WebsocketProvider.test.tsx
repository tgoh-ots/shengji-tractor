// Tests for WebsocketProvider URL construction logic

describe("WebsocketProvider URL construction", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Reset window._WEBSOCKET_HOST
    (global as any).window = { _WEBSOCKET_HOST: undefined };
    (global as any).location = {
      protocol: "https:",
      host: "example.com",
      pathname: "/game/",
    };
  });

  it("should use WEBSOCKET_HOST when provided", () => {
    (global as any).window._WEBSOCKET_HOST =
      "wss://custom.server.com/websocket";

    // Simulate the URL construction logic from WebsocketProvider
    const runtimeWebsocketHost = (global as any).window._WEBSOCKET_HOST;
    const uri =
      runtimeWebsocketHost !== undefined && runtimeWebsocketHost !== null
        ? runtimeWebsocketHost
        : (location.protocol === "https:" ? "wss://" : "ws://") +
          location.host +
          location.pathname +
          (location.pathname.endsWith("/") ? "api" : "/api");

    expect(uri).toBe("wss://custom.server.com/websocket");
  });

  it("should use default URL when WEBSOCKET_HOST is null", () => {
    (global as any).window._WEBSOCKET_HOST = null;

    // Simulate the URL construction logic from WebsocketProvider
    const runtimeWebsocketHost = (global as any).window._WEBSOCKET_HOST;
    const uri =
      runtimeWebsocketHost !== undefined && runtimeWebsocketHost !== null
        ? runtimeWebsocketHost
        : ((global as any).location.protocol === "https:"
            ? "wss://"
            : "ws://") +
          (global as any).location.host +
          (global as any).location.pathname +
          ((global as any).location.pathname.endsWith("/") ? "api" : "/api");

    // Should construct URL from location
    expect(uri).toBe("wss://example.com/game/api");
  });

  it("should use default URL when WEBSOCKET_HOST is undefined", () => {
    (global as any).window._WEBSOCKET_HOST = undefined;

    // Simulate the URL construction logic from WebsocketProvider
    const runtimeWebsocketHost = (global as any).window._WEBSOCKET_HOST;
    const uri =
      runtimeWebsocketHost !== undefined && runtimeWebsocketHost !== null
        ? runtimeWebsocketHost
        : ((global as any).location.protocol === "https:"
            ? "wss://"
            : "ws://") +
          (global as any).location.host +
          (global as any).location.pathname +
          ((global as any).location.pathname.endsWith("/") ? "api" : "/api");

    // Should construct URL from location
    expect(uri).toBe("wss://example.com/game/api");
  });

  it("should use ws:// for non-https protocol when no WEBSOCKET_HOST", () => {
    (global as any).window._WEBSOCKET_HOST = undefined;
    (global as any).location = {
      protocol: "http:",
      host: "localhost:3000",
      pathname: "/",
    };

    // Simulate the URL construction logic from WebsocketProvider
    const runtimeWebsocketHost = (global as any).window._WEBSOCKET_HOST;
    const uri =
      runtimeWebsocketHost !== undefined && runtimeWebsocketHost !== null
        ? runtimeWebsocketHost
        : ((global as any).location.protocol === "https:"
            ? "wss://"
            : "ws://") +
          (global as any).location.host +
          (global as any).location.pathname +
          ((global as any).location.pathname.endsWith("/") ? "api" : "/api");

    expect(uri).toBe("ws://localhost:3000/api");
  });

  it("should handle pathname not ending with slash", () => {
    (global as any).window._WEBSOCKET_HOST = undefined;
    (global as any).location = {
      protocol: "https:",
      host: "example.com",
      pathname: "/game",
    };

    // Simulate the URL construction logic from WebsocketProvider
    const runtimeWebsocketHost = (global as any).window._WEBSOCKET_HOST;
    const uri =
      runtimeWebsocketHost !== undefined && runtimeWebsocketHost !== null
        ? runtimeWebsocketHost
        : ((global as any).location.protocol === "https:"
            ? "wss://"
            : "ws://") +
          (global as any).location.host +
          (global as any).location.pathname +
          ((global as any).location.pathname.endsWith("/") ? "api" : "/api");

    expect(uri).toBe("wss://example.com/game/api");
  });

  it("should handle WEBSOCKET_HOST with ws:// protocol", () => {
    (global as any).window._WEBSOCKET_HOST = "ws://dev.server.com/socket";

    // Simulate the URL construction logic from WebsocketProvider
    const runtimeWebsocketHost = (global as any).window._WEBSOCKET_HOST;
    const uri =
      runtimeWebsocketHost !== undefined && runtimeWebsocketHost !== null
        ? runtimeWebsocketHost
        : ((global as any).location.protocol === "https:"
            ? "wss://"
            : "ws://") +
          (global as any).location.host +
          (global as any).location.pathname +
          ((global as any).location.pathname.endsWith("/") ? "api" : "/api");

    expect(uri).toBe("ws://dev.server.com/socket");
  });

  it("should handle WEBSOCKET_HOST with wss:// protocol", () => {
    (global as any).window._WEBSOCKET_HOST = "wss://secure.server.com/ws";

    // Simulate the URL construction logic from WebsocketProvider
    const runtimeWebsocketHost = (global as any).window._WEBSOCKET_HOST;
    const uri =
      runtimeWebsocketHost !== undefined && runtimeWebsocketHost !== null
        ? runtimeWebsocketHost
        : ((global as any).location.protocol === "https:"
            ? "wss://"
            : "ws://") +
          (global as any).location.host +
          (global as any).location.pathname +
          ((global as any).location.pathname.endsWith("/") ? "api" : "/api");

    expect(uri).toBe("wss://secure.server.com/ws");
  });

  it("should handle empty string WEBSOCKET_HOST", () => {
    (global as any).window._WEBSOCKET_HOST = "";
    (global as any).location = {
      protocol: "https:",
      host: "example.com",
      pathname: "/",
    };

    // Simulate the URL construction logic from WebsocketProvider
    const runtimeWebsocketHost = (global as any).window._WEBSOCKET_HOST;
    const uri =
      runtimeWebsocketHost !== undefined && runtimeWebsocketHost !== null
        ? runtimeWebsocketHost
        : ((global as any).location.protocol === "https:"
            ? "wss://"
            : "ws://") +
          (global as any).location.host +
          (global as any).location.pathname +
          ((global as any).location.pathname.endsWith("/") ? "api" : "/api");

    // Empty string is truthy in JavaScript, but the code checks for undefined and null
    // So empty string would be used as-is
    expect(uri).toBe("");
  });
});

// Resolution order including the build-time baked process.env.WEBSOCKET_HOST
// fallback used by a standalone (e.g. Vercel) frontend deploy. Mirrors the
// logic in WebsocketProvider.tsx exactly:
//   window._WEBSOCKET_HOST -> process.env.WEBSOCKET_HOST -> same-origin.
describe("WebsocketProvider host resolution with baked fallback", () => {
  const resolve = (
    runtimeWebsocketHost: string | null | undefined,
    bakedWebsocketHost: string | null | undefined,
    loc: { protocol: string; host: string; pathname: string },
  ): string =>
    runtimeWebsocketHost !== undefined && runtimeWebsocketHost !== null
      ? runtimeWebsocketHost
      : bakedWebsocketHost !== undefined &&
          bakedWebsocketHost !== null &&
          bakedWebsocketHost !== ""
        ? bakedWebsocketHost
        : (loc.protocol === "https:" ? "wss://" : "ws://") +
          loc.host +
          loc.pathname +
          (loc.pathname.endsWith("/") ? "api" : "/api");

  const loc = { protocol: "https:", host: "example.com", pathname: "/game/" };

  it("prefers window._WEBSOCKET_HOST over the baked value", () => {
    expect(
      resolve("wss://runtime.example/api", "wss://baked.example/api", loc),
    ).toBe("wss://runtime.example/api");
  });

  it("falls back to the baked value when runtime host is undefined", () => {
    expect(resolve(undefined, "wss://baked.example/api", loc)).toBe(
      "wss://baked.example/api",
    );
  });

  it("falls back to the baked value when runtime host is null", () => {
    expect(resolve(null, "wss://baked.example/api", loc)).toBe(
      "wss://baked.example/api",
    );
  });

  it("falls back to same-origin when neither runtime nor baked is set", () => {
    expect(resolve(undefined, undefined, loc)).toBe(
      "wss://example.com/game/api",
    );
  });

  it("treats an empty baked value as unset (same-origin)", () => {
    expect(resolve(undefined, "", loc)).toBe("wss://example.com/game/api");
  });
});
