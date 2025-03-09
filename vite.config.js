import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'
import { viteStaticCopy } from 'vite-plugin-static-copy'
import fs from 'fs'
import path from 'path'

const excludeFiles = [
  'blogposts/lessons-from-aws.md',
]

const getAllFiles = (dirPath, arrayOfFiles = []) => {
  const files = fs.readdirSync(dirPath)

  files.forEach((file) => {
    const fullPath = path.join(dirPath, file)
    if (fs.statSync(fullPath).isDirectory()) {
      getAllFiles(fullPath, arrayOfFiles)
    } else {
      const relativePath = path.relative('public', fullPath)
      if (!excludeFiles.includes(relativePath)) {
        arrayOfFiles.push(fullPath)
      }
    }
  })

  return arrayOfFiles
}

export default defineConfig({
  plugins: [
    react(),
    viteStaticCopy({
      targets: [
        {
          src: getAllFiles('public'),
          dest: '.',
          rename: (name, extension, fullPath) => {
            return path.relative('public', fullPath)
          }
        }
      ]
    })
  ],
  build: {
    outDir: 'dist',
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
      },
    },
  },
  base: '/',
  publicDir: false,
})
