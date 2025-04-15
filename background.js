let canvas = document.getElementById('backgroundCanvas');
let ctx = canvas ? canvas.getContext('2d') : null; // Get context only if canvas exists

let runBackground = true; // Flag to control the loop
const fixedOpacity = 0.15; // Opacity for the drawn points

// Global variable to store points for resize redraw
let currentPoints = [];


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
             // console.warn("Invalid point structure:", point); // Optional logging
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
            let pointsProxy = await generate();
            // Safely convert PyProxy to JS array and destroy proxy
            currentPoints = pointsProxy.toJs({ deep_proxies: false });
            pointsProxy.destroy();

            resizeCanvas(); // Ensure canvas size is current before drawing
            drawPoints(currentPoints);

            // --- Training Steps ---
            for (let i = 0; i < 400; i++) {
                if (!runBackground) break; // Check flag to allow stopping

                let boundaryProxy = await step(); // Perform one training step & get boundary
                let boundary = boundaryProxy.toJs({ deep_proxies: false });
                boundaryProxy.destroy();

                // Update points and redraw
                currentPoints = addBoundaryToPoints(currentPoints, boundary);
                drawPoints(currentPoints);
                await new Promise(resolve => setTimeout(resolve, 20)); // Short delay for rendering
            }

            if (!runBackground) break; // Check flag again after inner loop

            // Clear console periodically
            if (n % 100 === 0 && n > 0) {
                console.clear();
                console.log("Background animation running... (Console cleared)");
            }
        } // End of main loop

    } catch (error) {
        console.error("Error during background animation setup or loop:", error);
        // Decide how to handle errors (e.g., stop the animation)
        runBackground = false;
        // Optionally display a message to the user or just log
    } finally {
        console.log("Background animation loop finished or stopped.");
        // Cleanup if necessary (though Pyodide resources are managed internally)
    }
}

// --- Initialize ---
// Make sure the DOM is ready before trying to access canvas or run the main function
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        canvas = document.getElementById('backgroundCanvas');
        ctx = canvas ? canvas.getContext('2d') : null;
        if (runBackground && canvas) {
             mainBackgroundAnimation();
             window.addEventListener('resize', () => {
                resizeCanvas();
                drawPoints(currentPoints); // Redraw with current points on resize
             });
        } else if (!canvas) {
            console.warn("Background canvas not found on DOMContentLoaded.");
        }
    });
} else {
    // DOM is already ready
    canvas = document.getElementById('backgroundCanvas');
    ctx = canvas ? canvas.getContext('2d') : null;
     if (runBackground && canvas) {
         mainBackgroundAnimation();
         window.addEventListener('resize', () => {
            resizeCanvas();
            drawPoints(currentPoints); // Redraw with current points on resize
         });
    } else if (!canvas) {
        console.warn("Background canvas not found.");
    }
}

// Optional: Function to explicitly stop the animation if needed elsewhere
// function stopBackgroundAnimation() {
//     runBackground = false;
//     console.log("Stopping background animation loop.");
// }