# Agent Orchestrator TS

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![TypeScript](https://img.shields.io/badge/language-TypeScript-blue)

**Agent Orchestrator TS** is an open-source framework for orchestrating multiple AI agents in a structured and scalable way, entirely built in **TypeScript**. Inspired by [AWS Labs' Multi-Agent Orchestrator](https://github.com/awslabs/multi-agent-orchestrator), this project provides a flexible and developer-friendly environment for managing autonomous agents and their interactions.

## Features

- **TypeScript-first**: Fully written in TypeScript for better type safety and developer experience.
- **Extensible**: Easily integrate new agent types and workflows.
- **Event-driven architecture**: Use async messaging and event handling for seamless agent coordination.
- **Customizable**: Define agent behavior, interactions, and execution flows.
- **Open-source & community-driven**: Licensed under **Apache 2.0**, contributions are welcome!

## Getting Started

### Prerequisites

- Node.js 20.x or later
- npm or yarn

### Installation

```sh
npm install agent-orchestrator-ts
```

### Usage

```ts
import { MultiAgentOrchestrator, OpenAIAgent } from "agent-orchestrator-ts";

const OPENAI_MODEL = "gpt-4o-mini";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY!;
const AGENT_PROMPT = "You are an assistant in charge of ...";

const orchestrator = new MultiAgentOrchestrator();

const agent = new OpenAIAgent({
  name: `Agent's Name`,
  description: `Specialized agent in ...`,
  apiKey: OPENAI_API_KEY,
  model: OPENAI_MODEL,
  formatResponseAsJson: true,
});
agent.setSystemPrompt(AGENT_PROMPT);

orchestrator.addAgent(agent);

const response = await orchestrator.routeRequest("I want to ...", "userId", "sessionId");

// Handle the response (streaming or non-streaming)
if (response.streaming == true) {
  console.log("\n** RESPONSE STREAMING ** \n");
  // Send metadata immediately
  console.log(` > Agent ID: ${response.metadata.agentId}`);
  console.log(` > Agent Name: ${response.metadata.agentName}`);
  console.log(` > Agent Model: ${response.metadata.agentModel}`);
  console.log(`> User Input: ${response.metadata.userInput}`);
  console.log(`> User ID: ${response.metadata.userId}`);
  console.log(`> Session ID: ${response.metadata.sessionId}`);
  console.log(`> Additional Parameters:`, response.metadata.additionalParams);
  console.log(`\n> Response: `);
  for await (const chunk of response.output) {
    if (typeof chunk === "string") {
      process.stdout.write(chunk);
    } else {
      console.error("Received unexpected chunk type:", typeof chunk);
    }
  }
} else {
  // Handle non-streaming response (AgentProcessingResult)
  console.log("\n** RESPONSE ** \n");
  console.log(` > Agent ID: ${response.metadata.agentId}`);
  console.log(` > Agent Name: ${response.metadata.agentName}`);
  console.log(` > Agent Model: ${response.metadata.agentModel}`);
  console.log(`> User Input: ${response.metadata.userInput}`);
  console.log(`> User ID: ${response.metadata.userId}`);
  console.log(`> Session ID: ${response.metadata.sessionId}`);
  console.log(`> Additional Parameters:`, response.metadata.additionalParams);
  console.log(`\n> Response: ${response.output}`);
}
```

## Contributing

We welcome contributions from the community! Please check our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get started.

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

Inspired by the [AWS Labs Multi-Agent Orchestrator](https://github.com/awslabs/multi-agent-orchestrator), adapted for TypeScript.
