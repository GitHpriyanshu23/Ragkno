import React from 'react';
import {ProductUIRegistration} from './scenes/ProductUI';
import {Folder} from 'remotion';
import {LaunchRegistration} from './Launch';
import {OpeningRegistration} from './scenes/Opening';
import {IntroductionRegistration} from './scenes/Introduction';
import {SourcesRegistration} from './scenes/Sources';
import {ConversationRegistration} from './scenes/Conversation';
import {EvidenceRegistration} from './scenes/Evidence';
import {ClosingRegistration} from './scenes/Closing';
export function RemotionRoot() {
  return <><LaunchRegistration/><Folder name="Scenes"><OpeningRegistration/><IntroductionRegistration/><SourcesRegistration/><ConversationRegistration/><ProductUIRegistration/><EvidenceRegistration/><ClosingRegistration/></Folder></>;
}
