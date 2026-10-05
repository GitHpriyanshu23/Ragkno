import React from 'react';
import {AbsoluteFill,useCurrentFrame,interpolate} from 'remotion';
export function Wipe(){const f=useCurrentFrame();return <AbsoluteFill style={{pointerEvents:'none',overflow:'hidden'}}><div style={{position:'absolute',inset:-120,background:'#111111',translate:`${interpolate(f,[0,8,17],[-2300,0,2300])}px 0`,rotate:'-8deg'}}/><div style={{position:'absolute',inset:-120,background:'#3b82f6',translate:`${interpolate(f,[0,8,17],[-2600,0,2700])}px 0`,rotate:'-8deg'}}/></AbsoluteFill>}
