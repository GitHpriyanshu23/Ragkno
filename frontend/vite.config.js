import path from 'path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      // Proxy API calls to FastAPI during dev
      '/app-auth': 'http://localhost:8000',
      '/auth': 'http://localhost:8000',
      '/drive': 'http://localhost:8000',
      '/ingest': 'http://localhost:8000',
      '/memory': 'http://localhost:8000',
      '/query': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/threads': 'http://localhost:8000',
      '/feedback': 'http://localhost:8000',
    }
  }
})

