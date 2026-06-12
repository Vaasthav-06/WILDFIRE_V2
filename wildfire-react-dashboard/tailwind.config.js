/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0b0f19',
        panel: 'rgba(15, 23, 42, 0.7)',
        'panel-border': 'rgba(255, 255, 255, 0.1)',
        'risk-low': '#3b82f6',
        'risk-mod': '#fbbf24',
        'risk-high': '#ef4444',
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
