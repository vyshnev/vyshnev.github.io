// Get the button and the body elements
const darkModeToggle = document.getElementById('darkModeToggle');
const body = document.body;
const toggleIcon = darkModeToggle.querySelector('i'); // Get the icon inside the button

// Function to set the theme based on preference
const setTheme = (theme) => {
    if (!body || !toggleIcon || !darkModeToggle) {
        console.error("Dark mode elements not found");
        return;
    }
    body.classList.remove('light-mode', 'dark-mode'); // Remove existing classes first
    body.classList.add(theme + '-mode'); // Add the new theme class
    localStorage.setItem('theme', theme); // Save preference

    // Update icon based on theme
    if (theme === 'dark') {
        toggleIcon.classList.remove('fa-moon');
        toggleIcon.classList.add('fa-sun');
        darkModeToggle.title = "Switch to Light Mode";
    } else {
        toggleIcon.classList.remove('fa-sun');
        toggleIcon.classList.add('fa-moon');
         darkModeToggle.title = "Switch to Dark Mode";
    }
};

// Check for saved theme preference on page load
const currentTheme = localStorage.getItem('theme');

// Apply the saved theme or default to light
// Also consider system preference if no explicit choice is saved
if (currentTheme) {
    setTheme(currentTheme);
} else {
    // Optional: Check system preference
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    setTheme(prefersDark ? 'dark' : 'light');
    // Or just default to light:
    // setTheme('light');
}


// Add event listener to the toggle button only if it exists
if (darkModeToggle) {
    darkModeToggle.addEventListener('click', () => {
        // Check the *current* class on the body to determine the toggle target
        const themeToSet = body.classList.contains('dark-mode') ? 'light' : 'dark';
        setTheme(themeToSet);
    });
} else {
    console.warn("Dark mode toggle button not found.");
}

// Optional: Listen for system preference changes
try { // Wrap in try-catch for browsers that might not support addEventListener here
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', event => {
        // Only change if the user hasn't explicitly chosen a theme via localStorage
        if (!localStorage.getItem('theme')) {
            setTheme(event.matches ? 'dark' : 'light');
        }
    });
} catch (e) {
    console.warn("System theme change listener not supported or failed:", e);
}