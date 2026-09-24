import { useCallback, useEffect, useMemo, useRef } from 'react'
import { gsap } from 'gsap'
import './DotGrid.css'

const throttle = (func, limit) => {
  let lastCall = 0
  return function throttled(...args) {
    const now = performance.now()
    if (now - lastCall >= limit) {
      lastCall = now
      func.apply(this, args)
    }
  }
}

function hexToRgb(hex) {
  const match = hex.match(/^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i)
  if (!match) return { r: 0, g: 0, b: 0 }
  return {
    r: parseInt(match[1], 16),
    g: parseInt(match[2], 16),
    b: parseInt(match[3], 16),
  }
}

export default function DotGrid({
  dotSize = 16,
  gap = 32,
  baseColor = '#5227FF',
  activeColor = '#5227FF',
  proximity = 150,
  speedTrigger = 100,
  shockRadius = 250,
  shockStrength = 5,
  maxSpeed = 5000,
  resistance = 750,
  returnDuration = 1.5,
  className = '',
  style,
}) {
  const wrapperRef = useRef(null)
  const canvasRef = useRef(null)
  const dotsRef = useRef([])
  const pointerRef = useRef({
    x: -10000,
    y: -10000,
    vx: 0,
    vy: 0,
    speed: 0,
    lastTime: 0,
    lastX: 0,
    lastY: 0,
  })

  const baseRgb = useMemo(() => hexToRgb(baseColor), [baseColor])
  const activeRgb = useMemo(() => hexToRgb(activeColor), [activeColor])

  const circlePath = useMemo(() => {
    if (typeof window === 'undefined' || !window.Path2D) return null
    const path = new window.Path2D()
    path.arc(0, 0, dotSize / 2, 0, Math.PI * 2)
    return path
  }, [dotSize])

  const buildGrid = useCallback(() => {
    const wrap = wrapperRef.current
    const canvas = canvasRef.current
    if (!wrap || !canvas) return

    const { width, height } = wrap.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1

    canvas.width = width * dpr
    canvas.height = height * dpr
    canvas.style.width = `${width}px`
    canvas.style.height = `${height}px`

    const ctx = canvas.getContext('2d')
    if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    const cell = dotSize + gap
    const cols = Math.max(1, Math.floor((width + gap) / cell))
    const rows = Math.max(1, Math.floor((height + gap) / cell))
    const gridW = cell * cols - gap
    const gridH = cell * rows - gap
    const startX = (width - gridW) / 2 + dotSize / 2
    const startY = (height - gridH) / 2 + dotSize / 2

    const dots = []
    for (let y = 0; y < rows; y += 1) {
      for (let x = 0; x < cols; x += 1) {
        dots.push({
          cx: startX + x * cell,
          cy: startY + y * cell,
          xOffset: 0,
          yOffset: 0,
          _inertiaApplied: false,
        })
      }
    }
    dotsRef.current = dots
  }, [dotSize, gap])

  useEffect(() => {
    if (!circlePath) return undefined

    let rafId
    const proxSq = proximity * proximity

    const draw = () => {
      const canvas = canvasRef.current
      if (!canvas) return
      const ctx = canvas.getContext('2d')
      if (!ctx) return

      const dpr = window.devicePixelRatio || 1
      ctx.clearRect(0, 0, canvas.width / dpr, canvas.height / dpr)

      const { x: px, y: py } = pointerRef.current
      for (const dot of dotsRef.current) {
        const ox = dot.cx + dot.xOffset
        const oy = dot.cy + dot.yOffset
        const dx = dot.cx - px
        const dy = dot.cy - py
        const dsq = dx * dx + dy * dy

        let fill = baseColor
        if (dsq <= proxSq) {
          const dist = Math.sqrt(dsq)
          const t = 1 - dist / proximity
          const r = Math.round(baseRgb.r + (activeRgb.r - baseRgb.r) * t)
          const g = Math.round(baseRgb.g + (activeRgb.g - baseRgb.g) * t)
          const b = Math.round(baseRgb.b + (activeRgb.b - baseRgb.b) * t)
          fill = `rgb(${r},${g},${b})`
        }

        ctx.save()
        ctx.translate(ox, oy)
        ctx.fillStyle = fill
        ctx.fill(circlePath)
        ctx.restore()
      }

      rafId = requestAnimationFrame(draw)
    }

    draw()
    return () => cancelAnimationFrame(rafId)
  }, [activeRgb, baseColor, baseRgb, circlePath, proximity])

  useEffect(() => {
    buildGrid()
    let observer = null
    if ('ResizeObserver' in window) {
      observer = new ResizeObserver(buildGrid)
      if (wrapperRef.current) observer.observe(wrapperRef.current)
    } else {
      window.addEventListener('resize', buildGrid)
    }

    return () => {
      if (observer) observer.disconnect()
      else window.removeEventListener('resize', buildGrid)
    }
  }, [buildGrid])

  useEffect(() => {
    const releaseDot = (dot, delay = 0) => {
      gsap.to(dot, {
        xOffset: 0,
        yOffset: 0,
        duration: returnDuration,
        delay,
        ease: 'elastic.out(1,0.75)',
        onComplete: () => {
          dot._inertiaApplied = false
        },
      })
    }

    const pushDot = (dot, xOffset, yOffset, duration = 0.42) => {
      dot._inertiaApplied = true
      gsap.killTweensOf(dot)
      gsap.to(dot, {
        xOffset,
        yOffset,
        duration,
        ease: 'power3.out',
        onComplete: () => releaseDot(dot),
      })
    }

    const onMove = (event) => {
      const canvas = canvasRef.current
      if (!canvas) return

      const now = performance.now()
      const pointer = pointerRef.current
      const dt = pointer.lastTime ? now - pointer.lastTime : 16
      const dx = event.clientX - pointer.lastX
      const dy = event.clientY - pointer.lastY
      let vx = (dx / dt) * 1000
      let vy = (dy / dt) * 1000
      let speed = Math.hypot(vx, vy)

      if (speed > maxSpeed) {
        const scale = maxSpeed / speed
        vx *= scale
        vy *= scale
        speed = maxSpeed
      }

      pointer.lastTime = now
      pointer.lastX = event.clientX
      pointer.lastY = event.clientY
      pointer.vx = vx
      pointer.vy = vy
      pointer.speed = speed

      const rect = canvas.getBoundingClientRect()
      pointer.x = event.clientX - rect.left
      pointer.y = event.clientY - rect.top

      for (const dot of dotsRef.current) {
        const dist = Math.hypot(dot.cx - pointer.x, dot.cy - pointer.y)
        if (speed > speedTrigger && dist < proximity && !dot._inertiaApplied) {
          const force = Math.max(0, 1 - dist / proximity)
          const pushX = (dot.cx - pointer.x + vx * 0.006) * force
          const pushY = (dot.cy - pointer.y + vy * 0.006) * force
          pushDot(dot, pushX, pushY, Math.max(0.18, resistance / 2200))
        }
      }
    }

    const onClick = (event) => {
      const canvas = canvasRef.current
      if (!canvas) return

      const rect = canvas.getBoundingClientRect()
      const cx = event.clientX - rect.left
      const cy = event.clientY - rect.top

      for (const dot of dotsRef.current) {
        const dist = Math.hypot(dot.cx - cx, dot.cy - cy)
        if (dist < shockRadius && !dot._inertiaApplied) {
          const falloff = Math.max(0, 1 - dist / shockRadius)
          const pushX = (dot.cx - cx) * shockStrength * falloff
          const pushY = (dot.cy - cy) * shockStrength * falloff
          pushDot(dot, pushX, pushY, 0.34)
        }
      }
    }

    const throttledMove = throttle(onMove, 50)
    window.addEventListener('mousemove', throttledMove, { passive: true })
    window.addEventListener('click', onClick)

    return () => {
      window.removeEventListener('mousemove', throttledMove)
      window.removeEventListener('click', onClick)
    }
  }, [maxSpeed, proximity, resistance, returnDuration, shockRadius, shockStrength, speedTrigger])

  return (
    <div className={`dot-grid ${className}`} style={style}>
      <div ref={wrapperRef} className="dot-grid__wrap">
        <canvas ref={canvasRef} className="dot-grid__canvas" />
      </div>
    </div>
  )
}
