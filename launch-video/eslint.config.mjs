import { config } from "@remotion/eslint-config-flat";

export default [...config.map(rule => ({...rule, files:['src/**/*.jsx']}))];
