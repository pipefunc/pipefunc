module.exports = {
    testEnvironment: 'jsdom',  // Simulates a browser-like environment
    transform: {
      '^.+\\.js$': 'babel-jest',
    },
    setupFiles: ['./jest.setup.js'],
  };
