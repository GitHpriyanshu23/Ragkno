import React from 'react';
import {Composition,useCurrentFrame} from 'remotion';
import {Stage,Word,tween,pop,Pointer} from '../motion';
export function Evidence(){const f=useCurrentFrame();const morph=tween(f,57,77,0,1);return <Stage blue>
<Word style={{position:'absolute',top:145,width:'100%',textAlign:'center',fontSize:115,fontWeight:750,letterSpacing:-6}}>Built in the open.</Word>
<Word delay={10} style={{position:'absolute',top:290,width:'100%',textAlign:'center',fontSize:37}}>RagKno is an open-source application.</Word>
<div style={{position:'absolute',left:820,top:400,width:280,height:280,display:'grid',placeItems:'center',background:'#111111',borderRadius:65,scale:pop(f,15),boxShadow:'0 25px 70px #071d5140'}}>
<svg viewBox="0 0 24 24" width="165" height="165" style={{position:'absolute',fill:'#fff',opacity:1-morph,scale:1-morph*.7,rotate:`${morph*100}deg`}}><path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/></svg>
<svg viewBox="0 0 100 100" width="180" height="180" style={{position:'absolute',fill:'#ffd866',opacity:morph,scale:pop(f,61),rotate:`${tween(f,61,93,-100,0)}deg`}}><path d="M50 5L63 34L95 38L71 60L77 93L50 77L23 93L29 60L5 38L37 34Z"/></svg>
</div>
{[0,1,2,3,4,5,6,7].map(i=><div key={i} style={{position:'absolute',left:958+Math.cos(i*Math.PI/4)*tween(f,66,92,120,240),top:538+Math.sin(i*Math.PI/4)*tween(f,66,92,120,240),width:10,height:10,background:'#ffd866',borderRadius:5,opacity:tween(f,66,75,0,1)*(1-tween(f,80,103,0,1))}}/>)}
<Pointer x={tween(f,35,57,1270,1020)} y={tween(f,35,57,810,565)} click={f>=57&&f<63}/>
<Word delay={77} style={{position:'absolute',top:744,width:'100%',textAlign:'center',fontSize:61,fontWeight:700,letterSpacing:-2}}>Star us on GitHub.</Word>
<Word delay={87} style={{position:'absolute',top:838,width:'100%',textAlign:'center',fontSize:31}}>github.com/GitHpriyanshu23/Ragkno</Word>
</Stage>}
export const EvidenceRegistration=()=> <Composition id="Evidence" component={Evidence} durationInFrames={141} fps={30} width={1920} height={1080}/>;
