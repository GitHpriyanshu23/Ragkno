import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Proxy API calls to FastAPI during dev
      '/auth': 'http://localhost:8000',
      '/drive': 'http://localhost:8000',
      '/ingest': 'http://localhost:8000',
      '/memory': 'http://localhost:8000',
      '/query': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    }
  }
})
