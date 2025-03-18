import { Agent, AgentDescription, AgentOptions } from "./agent";
import {
  ConversationMessage,
  OPENAI_MODEL_ID_GPT_O_MINI,
  ParticipantRole,
  TemplateVariables,
} from "../types";
import OpenAI from "openai";
import { Logger } from "../utils/logger";
import { Retriever } from "../retrievers/retriever";

type WithApiKey = {
  apiKey: string;
  client?: never;
};

type WithClient = {
  client: OpenAI;
  apiKey?: never;
};

export interface OpenAIAgentOptions extends AgentOptions {
  model?: string;
  streaming?: boolean;
  inferenceConfig?: {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    stopSequences?: string[];
  };
  customSystemPrompt?: {
    template: string;
    variables?: TemplateVariables;
  };
  retriever?: Retriever;
  formatResponseAsJson?: boolean;
  toolConfig?: {
    tool: OpenAI.ChatCompletionTool[];
    useToolHandler: (response: any, conversation: any[]) => any;
    toolMaxRecursions?: number;
  };
}

export type OpenAIAgentOptionsWithAuth = OpenAIAgentOptions &
  (WithApiKey | WithClient);

const DEFAULT_MAX_TOKENS = 1000;

export enum OPEN_AI_MODELS_THAT_SUPPORT_FORMAT_RESPONSE_AS_JSON {
  GPT_4_5_PREVIEW = "gpt-4.5-preview",
  GPT_4_5_PREVIEW_2025_02_27 = "gpt-4.5-preview-2025-02-27",
  O3_MINI = "o3-mini",
  O3_MINI_2025_1_3 = "o3-mini-2025-1-3",
  O1 = "o1",
  O1_2024_12_17 = "o1-2024-12-17",
  GPT_4O_MINI = "gpt-4o-mini",
  GPT_4O_MINI_2024_07_18 = "gpt-4o-mini-2024-07-18",
  GPT_4O = "gpt-4o",
  GPT_4O_2024_08_06 = "gpt-4o-2024-08-06",
}

export class OpenAIAgent extends Agent {
  private client: OpenAI;
  private model: string;
  private streaming: boolean;
  private inferenceConfig: {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    stopSequences?: string[];
  };
  private promptTemplate: string;
  private systemPrompt: string;
  private customVariables: TemplateVariables;
  protected retriever?: Retriever;
  private formatResponseAsJson?: boolean;
  public toolConfig?: {
    tool: OpenAI.ChatCompletionTool[];
    useToolHandler: (response: any, conversation: any[]) => any;
    toolMaxRecursions?: number;
  };

  constructor(options: OpenAIAgentOptionsWithAuth) {
    super(options);

    if (!options.apiKey && !options.client) {
      throw new Error("OpenAI API key or OpenAI client is required");
    }
    if (options.client) {
      this.client = options.client;
    } else {
      if (!options.apiKey) throw new Error("OpenAI API key is required");
      this.client = new OpenAI({ apiKey: options.apiKey });
    }

    this.model = options.model ?? OPENAI_MODEL_ID_GPT_O_MINI;
    this.streaming = options.streaming ?? false;
    this.inferenceConfig = {
      maxTokens: options.inferenceConfig?.maxTokens ?? DEFAULT_MAX_TOKENS,
      temperature: options.inferenceConfig?.temperature,
      topP: options.inferenceConfig?.topP,
      stopSequences: options.inferenceConfig?.stopSequences,
    };

    this.retriever = options.retriever ?? null;
    this.toolConfig = options.toolConfig ?? null;

    this.promptTemplate = `You are a ${this.name}. ${this.description} Provide helpful and accurate information based on your expertise.
    You will engage in an open-ended conversation, providing helpful and accurate information based on your expertise.
    The conversation will proceed as follows:
    - The human may ask an initial question or provide a prompt on any topic.
    - You will provide a relevant and informative response.
    - The human may then follow up with additional questions or prompts related to your previous response, allowing for a multi-turn dialogue on that topic.
    - Or, the human may switch to a completely new and unrelated topic at any point.
    - You will seamlessly shift your focus to the new topic, providing thoughtful and coherent responses based on your broad knowledge base.
    Throughout the conversation, you should aim to:
    - Understand the context and intent behind each new question or prompt.
    - Provide substantive and well-reasoned responses that directly address the query.
    - Draw insights and connections from your extensive knowledge when appropriate.
    - Ask for clarification if any part of the question or prompt is ambiguous.
    - Maintain a consistent, respectful, and engaging tone tailored to the human's communication style.
    - Seamlessly transition between topics as the human introduces new subjects.`;

    this.customVariables = {};
    this.systemPrompt = "";

    if (options.customSystemPrompt) {
      this.setSystemPrompt(
        options.customSystemPrompt.template,
        options.customSystemPrompt.variables
      );
    }

    if (options.formatResponseAsJson) {
      const models = Object.values(OPEN_AI_MODELS_THAT_SUPPORT_FORMAT_RESPONSE_AS_JSON);
      if (
        !models.includes(
          this.model as OPEN_AI_MODELS_THAT_SUPPORT_FORMAT_RESPONSE_AS_JSON
        )
      ) {
        throw new Error(`model must be one of  ${models.join(", ")}`);
      }

      this.formatResponseAsJson = options.formatResponseAsJson;
    }
  }

  /* eslint-disable @typescript-eslint/no-unused-vars */
  async processRequest(
    inputText: string,
    userId: string,
    sessionId: string,
    chatHistory: ConversationMessage[],
    additionalParams?: Record<string, string>
  ): Promise<ConversationMessage | AsyncIterable<any>> {
    this.updateSystemPrompt();

    let systemPrompt = this.systemPrompt;

    if (this.retriever) {
      // retrieve from Vector store
      const response =
        await this.retriever.retrieveAndCombineResults(inputText);
      const contextPrompt =
        "\nHere is the context to use to answer the user's question:\n" +
        response;
      systemPrompt = systemPrompt + contextPrompt;
    }

    const messages = [
      { role: "system", content: systemPrompt },
      ...chatHistory.map((msg) => ({
        role: msg.role.toLowerCase() as OpenAI.Chat.ChatCompletionMessageParam["role"],
        content: msg.content[0]?.text || "",
      })),
      { role: "user" as const, content: inputText },
    ] as OpenAI.Chat.ChatCompletionMessageParam[];

    const { maxTokens, temperature, topP, stopSequences } =
      this.inferenceConfig;

    const requestOptions: OpenAI.Chat.ChatCompletionCreateParams = {
      model: this.model,
      messages: messages,
      max_tokens: maxTokens,
      stream: this.streaming,
      temperature,
      top_p: topP,
      stop: stopSequences,
      response_format: {
        type: this.formatResponseAsJson ? "json_object" : "text",
      },
      tools: this.toolConfig?.tool || undefined,
    };

    if (this.streaming) {
      return this.handleStreamingResponse(requestOptions);
    } else {
      let finalMessage: string = "";
      let toolUse = false;
      let recursions = this.toolConfig?.toolMaxRecursions || 5;

      do {
        const response = await this.handleSingleResponse(requestOptions);

        if (response.tool_calls) {
          messages.push(response);

          if (!this.toolConfig) {
            throw new Error("No tools available for tool use");
          }

          const toolResponse = await this.toolConfig.useToolHandler(
            response,
            messages
          );
          messages.push(toolResponse);
          toolUse = true;
        } else {
          finalMessage = response.content;
          toolUse = false;
        }

        recursions--;
      } while (toolUse && recursions > 0);

      return {
        role: ParticipantRole.ASSISTANT,
        content: [{ text: finalMessage }],
      };
    }
  }

  setSystemPrompt(template?: string, variables?: TemplateVariables): void {
    if (template) {
      this.promptTemplate = template;
    }
    if (variables) {
      this.customVariables = variables;
    }
    this.updateSystemPrompt();
  }

  private updateSystemPrompt(): void {
    const allVariables: TemplateVariables = {
      ...this.customVariables,
    };
    this.systemPrompt = this.replaceplaceholders(
      this.promptTemplate,
      allVariables
    );
  }

  private replaceplaceholders(
    template: string,
    variables: TemplateVariables
  ): string {
    return template.replace(/{{(\w+)}}/g, (match, key) => {
      if (key in variables) {
        const value = variables[key];
        return Array.isArray(value) ? value.join("\n") : String(value);
      }
      return match;
    });
  }

  private async handleSingleResponse(
    input: any
  ): Promise<OpenAI.Chat.ChatCompletionMessage> {
    try {
      const nonStreamingOptions = { ...input, stream: false };
      const chatCompletion =
        await this.client.chat.completions.create(nonStreamingOptions);
      if (!chatCompletion.choices || chatCompletion.choices.length === 0) {
        throw new Error("No choices returned from OpenAI API");
      }

      const assistantMessage = chatCompletion.choices[0]?.message?.content;

      console.log(`Response from OpenAI choices: ${JSON.stringify(chatCompletion.choices)}`);
      console.log(`Response from OpenAI choices[0].message: ${JSON.stringify(chatCompletion.choices[0].message)}`);
      console.log(`Response from OpenAI chatCompletion: ${JSON.stringify(chatCompletion)}`);

      if (typeof assistantMessage !== "string") {
        throw new Error("Unexpected response format from OpenAI API");
      }

      const message = chatCompletion.choices[0].message;
      return message as OpenAI.Chat.ChatCompletionMessage;
    } catch (error) {
      Logger.logger.error("Error in OpenAI API call:", error);
      throw error;
    }
  }

  private async *handleStreamingResponse(
    options: OpenAI.Chat.ChatCompletionCreateParams
  ): AsyncIterable<string> {
    let recursions = this.toolConfig?.toolMaxRecursions || 5;

    while (recursions > 0) {
      // Add tool calls to messages before creating stream
      const messagesWithToolCalls = [...options.messages];

      const stream = await this.client.chat.completions.create({
        ...options,
        messages: messagesWithToolCalls,
        stream: true,
      });

      let currentToolCalls: any[] = [];
      let hasToolCalls = false;

      for await (const chunk of stream) {
        const toolCalls = chunk.choices[0]?.delta?.tool_calls;

        if (toolCalls) {
          for (const toolCall of toolCalls) {
            if (toolCall.id) {
              currentToolCalls.push({
                id: toolCall.id,
                function: toolCall.function,
              });
            }

            if (toolCall.function?.arguments) {
              const lastToolCall =
                currentToolCalls[currentToolCalls.length - 1];
              lastToolCall.function.arguments =
                (lastToolCall.function.arguments || "") +
                toolCall.function.arguments;
            }
          }
        }

        if (chunk.choices[0]?.finish_reason === "tool_calls") {
          hasToolCalls = true;
          const toolCallResults = [];

          // Add tool calls to messages before processing
          messagesWithToolCalls.push({
            role: "assistant",
            tool_calls: currentToolCalls.map((tc) => ({
              id: tc.id,
              type: "function",
              function: tc.function,
            })),
          });

          for (const toolCall of currentToolCalls) {
            try {
              const toolResponse = await this.toolConfig.useToolHandler(
                { tool_calls: [toolCall] },
                messagesWithToolCalls
              );

              toolCallResults.push({
                role: "tool",
                tool_call_id: toolCall.id,
                content: JSON.stringify(toolResponse),
              });
            } catch (error) {
              console.error("Tool call error", error);
            }
          }

          // Append tool call results to messages
          messagesWithToolCalls.push(...toolCallResults);

          // Update options for next iteration
          options.messages = messagesWithToolCalls;

          currentToolCalls = [];
        }

        const content = chunk.choices[0]?.delta?.content;
        if (content) {
          yield content;
        }
      }

      // Break if no tool calls were found
      if (!hasToolCalls) break;

      recursions--;
    }
  }

  getModel(): string {
    return this.model;
  }

  getChilds(): { [key: string]: AgentDescription } {
    return {};
  }
}
