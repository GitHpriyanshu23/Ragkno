import React from 'react';
import {Html5Audio as Audio} from 'remotion';
import {AbsoluteFill, Composition, Sequence, staticFile} from 'remotion';
import {Wipe} from './Wipe';
import {Opening} from './scenes/Opening';
import {Introduction} from './scenes/Introduction';
import {Sources} from './scenes/Sources';
import {Conversation} from './scenes/Conversation';
import {ProductUI} from './scenes/ProductUI';
import {Evidence} from './scenes/Evidence';
import {Closing} from './scenes/Closing';

export function Launch() {
  return <AbsoluteFill style={{background:'#111111'}}>
    <Audio name="Electronic soundtrack + synchronized clicks, typing, swooshes and success tones" src={staticFile('media/launch-electronic-soundtrack-v2.mp3')} volume={1}/>
    <Sequence name="Stop searching. Start asking." durationInFrames={112}><Opening/></Sequence>
    <Sequence name="Files converge · RagKno reveal" from={112} durationInFrames={100}><Introduction/></Sequence>
    <Sequence name="Animated document drop · indexing" from={212} durationInFrames={123}><Sources/></Sequence>
    <Sequence name="Question → answer → source excerpt" from={335} durationInFrames={190}><Conversation/></Sequence>
    <Sequence name="RagKno homepage · get started" from={525} durationInFrames={100}><ProductUI/></Sequence>
    <Sequence name="Open source · GitHub transforms into a star" from={625} durationInFrames={160}><Evidence/></Sequence>
    <Sequence name="RagKno.com · launch card" from={785} durationInFrames={115}><Closing/></Sequence>
    <Sequence name="Ink transition 1" from={103} durationInFrames={18}><Wipe/></Sequence>
    <Sequence name="Ink transition 2" from={203} durationInFrames={18}><Wipe/></Sequence>
    <Sequence name="Ink transition 3" from={326} durationInFrames={18}><Wipe/></Sequence>
    <Sequence name="Blue transition" from={616} durationInFrames={18}><Wipe/></Sequence>
    <Sequence name="Final reveal" from={776} durationInFrames={18}><Wipe/></Sequence>
  </AbsoluteFill>;
}
export const LaunchRegistration=()=> <Composition id="RagKno-Launch" component={Launch} durationInFrames={900} fps={30} width={1920} height={1080} defaultProps={{}}/>;
