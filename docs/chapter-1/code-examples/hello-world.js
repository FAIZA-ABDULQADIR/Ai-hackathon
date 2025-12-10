// Simple "Hello, World!" example for Docusaurus
function helloWorld() {
  console.log("Hello, World!");
  return "Hello, World!";
}

// Example usage
const message = helloWorld();
console.log("The function returned:", message);

// Export the function for use in other modules
module.exports = { helloWorld };