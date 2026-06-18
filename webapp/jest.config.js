const nextJest = require("next/jest");

const createJestConfig = nextJest({
  // Points next/jest at the Next.js app root so it loads next.config.js
  // and .env.local correctly during tests.
  dir: "./",
});

/** @type {import('jest').Config} */
const customConfig = {
  testEnvironment: "jest-environment-jsdom",
  // Resolve @/* path aliases defined in tsconfig.json
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/$1",
  },
  testMatch: ["**/__tests__/**/*.test.ts", "**/__tests__/**/*.test.tsx"],
};

module.exports = createJestConfig(customConfig);
