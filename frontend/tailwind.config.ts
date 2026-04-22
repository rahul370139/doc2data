import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo"],
      },
      colors: {
        ink: {
          50: "#F6F7F9",
          100: "#ECEEF2",
          200: "#D6DAE2",
          300: "#B3B9C6",
          400: "#858DA1",
          500: "#5A6379",
          600: "#41495C",
          700: "#2E3545",
          800: "#1D2231",
          900: "#11131C",
        },
        accent: {
          50: "#EEF6FF",
          100: "#D5E8FF",
          200: "#A9D0FF",
          300: "#71B0FF",
          400: "#3B8EFF",
          500: "#1F6FEB",
          600: "#1756C2",
          700: "#134396",
          800: "#0F346F",
          900: "#0B264E",
        },
        ok: { 500: "#16A34A", 100: "#DCFCE7" },
        warn: { 500: "#D97706", 100: "#FEF3C7" },
        bad: { 500: "#DC2626", 100: "#FEE2E2" },
      },
      boxShadow: {
        card: "0 1px 2px rgb(0 0 0 / 0.04), 0 4px 16px rgb(0 0 0 / 0.06)",
      },
      borderRadius: { xl2: "1.25rem" },
      animation: {
        pulseDot: "pulseDot 1.4s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
      keyframes: {
        pulseDot: {
          "0%, 100%": { opacity: "0.4" },
          "50%": { opacity: "1" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
