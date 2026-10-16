/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        clinical: {
          bg: '#f8fafc',
          surface: '#ffffff',
          card: '#ffffff',
          border: '#e2e8f0',
          muted: '#64748b',
          charcoal: '#0f172a',
          body: '#334155',
          teal: '#0d9488',
          tealDark: '#0f766e',
          tealLight: '#f0fdfa',
          tealBorder: '#ccfbf1',
          emerald: '#059669',
          amber: '#d97706',
          rose: '#e11d48',
        }
      },
      fontFamily: {
        sans: [
          'Inter',
          '"Open Sans"',
          'Roboto',
          '-apple-system',
          'BlinkMacSystemFont',
          'sans-serif',
        ],
        inter: ['Inter', 'sans-serif'],
        'open-sans': ['"Open Sans"', 'sans-serif'],
        roboto: ['Roboto', 'sans-serif'],
        'source-sans': ['"Source Sans 3"', '"Source Sans Pro"', 'sans-serif'],
        lato: ['Lato', 'sans-serif'],
        proxima: ['"Proxima Nova"', 'Inter', '-apple-system', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
