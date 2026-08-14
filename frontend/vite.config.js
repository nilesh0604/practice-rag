import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Vite config for the CTC-RAG chat widget.
// Dev server proxies /api to the FastAPI backend (default :8000).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
});
