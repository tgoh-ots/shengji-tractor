const path = require("path");
const webpack = require("webpack");
const TerserJsPlugin = require("terser-webpack-plugin");
const WasmPackPlugin = require("@wasm-tool/wasm-pack-plugin");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const CopyWebpackPlugin = require("copy-webpack-plugin");
const MiniCssExtractPlugin = require("mini-css-extract-plugin");
const CssMinimizerPlugin = require("css-minimizer-webpack-plugin");

// Optional build-time WebSocket host, baked into the bundle for standalone
// (e.g. Vercel) frontend deploys where the backend's /runtime.js is not served.
// When unset (the single-service Koyeb deploy), this is an empty string and the
// app falls back to window._WEBSOCKET_HOST (from /runtime.js) and then to
// same-origin. See WebsocketProvider.tsx / WasmOrRpcProvider.tsx.
const WEBSOCKET_HOST = process.env.WEBSOCKET_HOST || "";

module.exports = {
  mode: "production",
  devtool: "source-map",
  resolve: {
    extensions: [".ts", ".tsx", ".js", ".jsx", ".svg"],
  },
  entry: "./src/index.tsx",
  experiments: {
    asyncWebAssembly: true,
  },
  module: {
    rules: [
      {
        test: /\.ts(x?)$/,
        exclude: /node_modules/,
        use: [
          {
            loader: "ts-loader",
          },
        ],
      },
      {
        test: /\.css$/i,
        use: [
          MiniCssExtractPlugin.loader,
          "css-loader",
          {
            loader: "postcss-loader",
            options: {
              postcssOptions: {
                config: path.resolve(__dirname, "postcss.config.js"),
              },
            },
          },
        ],
      },
    ],
  },
  optimization: {
    moduleIds: "deterministic",
    splitChunks: {
      chunks: "all",
      cacheGroups: {
        cards: {
          test: /([\\/]playing-cards(-4color)?[\\/])|(SvgCard.tsx)/,
          name(_) {
            return "playing-cards";
          },
        },
      },
    },
    minimizer: [
      (compiler) => () => {
        new TerserJsPlugin({ terserOptions: { sourceMap: true } }).apply(
          compiler
        );
      },
      new CssMinimizerPlugin({}),
    ],
  },
  output: {
    filename: "[name].[contenthash].js",
  },
  performance: {
    hints: false,
  },
  plugins: [
    new webpack.DefinePlugin({
      "process.env.WEBSOCKET_HOST": JSON.stringify(WEBSOCKET_HOST),
    }),
    new WasmPackPlugin({
      crateDirectory: path.resolve(__dirname, "shengji-wasm"),
      outName: "shengji-core",
    }),
    new HtmlWebpackPlugin({
      filename: "index.html",
      template: "static/index.html",
    }),
    new MiniCssExtractPlugin({
      filename: "style.css",
    }),
    new CopyWebpackPlugin({
      patterns: [
        {
          from: "static/rules.html",
          to: "rules.html",
        },
        {
          from: "static/timer-worker.js",
          to: "timer-worker.js",
        },
        {
          from: "static/434472_dersuperanton_taking-card.mp3",
          to: "434472_dersuperanton_taking-card.mp3",
        },
      ],
    }),
  ],
};
