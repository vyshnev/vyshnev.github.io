let canvas = document.getElementById('backgroundCanvas');
let ctx = canvas ? canvas.getContext('2d') : null; // Get context only if canvas exists

let runBackground = true; // Flag to control the loop
const fixedOpacity = 0.15; // Opacity for the drawn points

// Global variable to store points for resize redraw
let currentPoints = []; // This will now hold the direct JS array

function resizeCanvas() {
    if (!canvas) return;
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}

function drawPoints(points) {
    if (!ctx || !canvas) return; // Ensure canvas and context are available
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const minDimension = Math.min(canvas.width, canvas.height);

    if (!points || !Array.isArray(points)) return; // Check if points is a valid array

    points.forEach(point => {
        // Basic validation for each point structure
        if (!Array.isArray(point) || point.length < 3 || typeof point[0] !== 'number' || typeof point[1] !== 'number' || typeof point[2] !== 'number') {
             // console.warn("Skipping invalid point structure:", point); // Optional logging
             return; // Skip invalid points
        }

        ctx.beginPath();
        ctx.arc(
            point[0] * canvas.width,  // X coordinate
            point[1] * canvas.height, // Y coordinate
            0.005 * minDimension,     // Radius
            0,
            Math.PI * 2
        );
        let color = ``;
        // Use fixedOpacity defined above
        switch (point[2]) { // Use switch for clarity
            case 0: // Label 0 (Blue)
                 color = `rgba(0,0,255,${fixedOpacity})`;
                 break;
            case 1: // Label 1 (Red)
                 color = `rgba(255,0,0,${fixedOpacity})`;
                 break;
            case 2: // Label 2 (Black boundary)
                 color = `rgba(0,0,0,${fixedOpacity})`;
                 break;
            default: // Default grey for unexpected labels
                 color = `rgba(128,128,128,${fixedOpacity})`;
        }
        ctx.fillStyle = color;
        ctx.fill();
    });
}

function addBoundaryToPoints(points, boundary) {
    // Ensure inputs are valid arrays
    if (!Array.isArray(points)) points = [];
    if (!Array.isArray(boundary)) boundary = [];

    // Filter out old boundary points (label 2)
    let filteredPoints = points.filter(point => Array.isArray(point) && point.length >=3 && point[2] !== 2);

    // Map new boundary points to the required format [x, y, label=2]
    let newBoundaryPoints = boundary.map(point => {
        if (Array.isArray(point) && point.length >= 2 && typeof point[0] === 'number' && typeof point[1] === 'number') {
            return [point[0], point[1], 2];
        }
        return null; // Invalid boundary point format
    }).filter(p => p !== null); // Remove any nulls

    return filteredPoints.concat(newBoundaryPoints);
}

async function mainBackgroundAnimation() {
    // Defensive checks at the start
    if (typeof loadPyodide !== 'function') {
        console.error("Pyodide library not loaded.");
        return; // Stop if pyodide isn't available
    }
     if (!canvas || !ctx) {
        console.error("Canvas element or context not found for background animation.");
        return; // Stop if canvas isn't available
    }

    try {
        console.log("Loading Pyodide...");
        let pyodide = await loadPyodide();
        console.log("Loading NumPy...");
        await pyodide.loadPackage("numpy");
        console.log("Fetching Python script (nn.py)...");
        let script = await fetch('nn.py').then(response => {
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return response.text();
        });
        console.log("Running Python script...");
        pyodide.runPython(script);

        // Get Python functions after script execution
        let generate = pyodide.globals.get("generate");
        let initialize_model = pyodide.globals.get("initialize_model");
        let step = pyodide.globals.get("step");

        // Check if functions were loaded correctly
        if (typeof generate !== 'function' || typeof initialize_model !== 'function' || typeof step !== 'function') {
            throw new Error("Failed to load Python functions (generate, initialize_model, step).");
        }


        console.log("Initializing NN model...");
        initialize_model();

        console.log("Starting background animation loop.");
        resizeCanvas(); // Initial resize

        for (let n = 0; n < 999999 && runBackground; n++) { // Main loop
            // --- Data Generation ---
            // Directly get the JavaScript array, as nn.py uses to_js()
            let generatedData = await generate();

            // Check if the returned data is a valid array
            if (!Array.isArray(generatedData)) {
                console.error("Error: generate() did not return a valid Array. Skipping epoch.", generatedData);
                await new Promise(resolve => setTimeout(resolve, 1000)); // Wait before retrying
                continue; // Skip to the next iteration of the outer loop
            }
            currentPoints = generatedData; // Assign directly

            resizeCanvas(); // Ensure canvas size is current before drawing
            drawPoints(currentPoints);

            // --- Training Steps ---
            for (let i = 0; i < 400; i++) {
                if (!runBackground) break; // Check flag to allow stopping

                 // Directly get the JavaScript array for the boundary
                let boundaryData = await step();

                // Check if the returned data is a valid array
                if (!Array.isArray(boundaryData)) {
                     console.error("Error: step() did not return a valid Array. Skipping step.", boundaryData);
                     await new Promise(resolve => setTimeout(resolve, 50)); // Short wait
                    continue; // Skip to next iteration of inner loop
                }

                // Update points and redraw (no .toJs or destroy needed)
                currentPoints = addBoundaryToPoints(currentPoints, boundaryData);
                drawPoints(currentPoints);
                await new Promise(resolve => setTimeout(resolve, 20)); // Short delay for rendering
            } // End of inner loop

            if (!runBackground) break; // Check flag again after inner loop

            // Clear console periodically
            if (n > 0 && n % 100 === 0 ) { // Clear every 100 epochs
                console.clear();
                console.log("Background animation running... (Console cleared)");
            }
        } // End of main loop

    } catch (error) {
        console.error("CRITICAL Error during background animation setup or loop:", error);
        runBackground = false; // Stop the animation on critical error
    } finally {
        console.log("Background animation loop finished or stopped.");
    }
}

// --- Initialize ---
// Use DOMContentLoaded to ensure elements are ready
function initializeApp() {
    canvas = document.getElementById('backgroundCanvas');
    ctx = canvas ? canvas.getContext('2d') : null;
    if (runBackground && canvas) {
         console.log("DOM ready, starting background animation.")
         mainBackgroundAnimation(); // Start the main async function
         window.addEventListener('resize', () => {
            resizeCanvas();
            // Redraw immediately on resize with the last known points
            drawPoints(currentPoints);
         });
    } else if (!canvas) {
        console.warn("Background canvas element not found.");
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    // DOM is already ready
    initializeApp();
}