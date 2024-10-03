// jest.config.js
module.exports = {
  testEnvironment: 'jsdom',
  transform: {
    '^.+\\.js$': 'babel-jest',
  },
  setupFiles: ['./jest.setup.js'],
  moduleNameMapper: {
    // Map CSS and other file types if needed
  },
  testMatch: ['**/src/**/*.test.js'], // Ensure Jest looks for tests in the 'src' directory
};
