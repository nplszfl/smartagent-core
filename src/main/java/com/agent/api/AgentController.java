package com.agent.api;

import com.agent.core.memory.Memory;
import com.agent.core.model.ChatModel;
import com.agent.core.model.ModelResponse;
import com.agent.core.tool.*;
import com.agent.entity.AgentMetricsRecord;
import com.agent.entity.ConversationEntity;
import com.agent.service.AgentMetricsService;
import com.agent.service.AgentTemplateService;
import com.agent.service.ConversationService;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import jakarta.validation.Valid;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Agent REST API Controller
 */
@Slf4j
@RestController
@RequestMapping("/api/agent")
public class AgentController {

    private final ChatModel chatModel;
    private final ToolRegistry toolRegistry;
    private final ConversationService conversationService;
    private final ObjectMapper objectMapper;

    @Autowired(required = false)
    private AgentMetricsService metricsService;

    @Autowired(required = false)
    private AgentTemplateService templateService;

    public AgentController(ChatModel chatModel, ConversationService conversationService) {
        this.chatModel = chatModel;
        this.conversationService = conversationService;
        this.toolRegistry = new ToolRegistry();
        // 注册默认工具
        toolRegistry.register(new SearchTool());
        toolRegistry.register(new CalculatorTool());
        toolRegistry.register(new DateTimeTool());
        toolRegistry.register(new OcrTool());
        this.objectMapper = new ObjectMapper();
        this.objectMapper.findAndRegisterModules();
    }

    /**
     * Chat with Agent
     */
    @PostMapping("/chat")
    public AgentResponse chat(@Valid @RequestBody AgentRequest request) {
        long startMs = System.currentTimeMillis();
        String userId = getUserId(request);
        String input = request.getInput();
        String agentType = request.getAgentType() != null ? request.getAgentType() : "react";
        String agentName = request.getAgentName() != null ? request.getAgentName() : "assistant";
        boolean success = false;
        boolean timeout = false;
        int iterations = 0;
        int inTokens = 0;
        int outTokens = 0;
        Set<String> toolsUsed = new LinkedHashSet<>();
        String errorMessage = null;

        try {
            int maxIterations = request.getMaxIterations() != null ? request.getMaxIterations() : 10;
            if (maxIterations <= 0) maxIterations = 10;

            // 解析系统提示：优先用模板
            String systemPrompt = request.getSystemPrompt();
            if (systemPrompt == null && templateService != null) {
                Optional<com.agent.entity.AgentTemplate> tmpl = templateService.getById(agentName);
                if (tmpl.isPresent()) {
                    systemPrompt = tmpl.get().getSystemPrompt();
                    templateService.recordUsage(agentName);
                }
            }
            if (systemPrompt == null) systemPrompt = buildDefaultSystemPrompt();

            // 限定工具列表
            List<Tool> activeTools = selectTools(request.getTools());

            // 获取或创建会话
            ConversationEntity conversation = conversationService.getOrCreateConversation(
                userId, agentName, systemPrompt);

            // 从数据库加载消息历史
            List<Memory.Message> messages = loadMessages(conversation);

            // 添加新用户消息
            if (request.getImageBase64() != null && !request.getImageBase64().isEmpty()) {
                List<Memory.ContentItem> contentItems = new ArrayList<>();
                if (input != null && !input.isEmpty()) {
                    contentItems.add(new Memory.ContentItem.Text(input));
                }
                String pureBase64 = extractPureBase64(request.getImageBase64());
                String format = extractImageFormat(request.getImageName());
                contentItems.add(new Memory.ContentItem.Image(pureBase64, format));
                messages.add(new Memory.Message(Memory.Message.Role.USER, contentItems));
            } else {
                messages.add(new Memory.Message(Memory.Message.Role.USER, input));
            }

            // 执行 ReAct 循环
            ChatLoopResult result = executeReActLoop(messages, activeTools, maxIterations);
            iterations = result.iterations;
            toolsUsed.addAll(result.toolsUsed);
            if (result.timedOut) timeout = true;

            // 更新数据库中的消息历史（保存失败不能阻塞聊天响应，但要记录到 metrics）
            try {
                saveMessages(conversation.getId(), messages);
            } catch (Exception saveEx) {
                log.error("保存消息历史失败（用户答案已返回，但会话历史可能丢失）", saveEx);
                errorMessage = "保存会话历史失败: " + saveEx.getMessage();
                success = false;
                return AgentResponse.success(result.answer);
            }

            success = true;
            return AgentResponse.success(result.answer);
        } catch (Exception e) {
            log.error("Chat处理失败", e);
            errorMessage = e.getMessage();
            return AgentResponse.error(errorMessage);
        } finally {
            recordMetrics(userId, agentName, agentType, input, success, timeout, errorMessage,
                System.currentTimeMillis() - startMs, iterations, inTokens, outTokens, toolsUsed);
        }
    }

    /**
     * 根据 tools 字段限定可用工具
     */
    private List<Tool> selectTools(List<String> tools) {
        if (tools == null || tools.isEmpty()) {
            return toolRegistry.getAll();
        }
        List<Tool> selected = new ArrayList<>();
        for (String name : tools) {
            Tool t = toolRegistry.get(name);
            if (t != null) selected.add(t);
        }
        return selected.isEmpty() ? toolRegistry.getAll() : selected;
    }

    /**
     * 记录调用指标
     */
    private void recordMetrics(String userId, String agentName, String agentType, String input,
                                boolean success, boolean timeout, String error,
                                long durationMs, int iterations, int inTokens, int outTokens,
                                Set<String> toolsUsed) {
        if (metricsService == null) return;
        try {
            AgentMetricsRecord record = AgentMetricsRecord.builder()
                .userId(userId)
                .agentName(agentName)
                .agentType(agentType)
                .input(input != null && input.length() > 500 ? input.substring(0, 500) : input)
                .success(success)
                .timeout(timeout)
                .error(error)
                .durationMs(durationMs)
                .iterations(iterations)
                .inputTokens(inTokens)
                .outputTokens(outTokens)
                .totalTokens(inTokens + outTokens)
                .toolsUsed(toolsUsed.stream().collect(Collectors.joining(",")))
                .modelName(chatModel != null ? chatModel.getModelName() : "unknown")
                .createdAt(LocalDateTime.now())
                .build();
            metricsService.record(record);
        } catch (Exception e) {
            log.warn("记录指标失败: {}", e.getMessage());
        }
    }

    /**
     * 获取用户会话列表
     */
    @GetMapping("/conversations")
    public List<ConversationEntity> getConversations(@RequestParam String userId) {
        return conversationService.getUserConversations(userId);
    }

    /**
     * 获取单个会话详情
     */
    @GetMapping("/conversation/{id}")
    public AgentResponse getConversation(@PathVariable Long id) {
        ConversationEntity conv = conversationService.getConversation(id);
        if (conv == null) {
            return AgentResponse.error("会话不存在");
        }
        try {
            return AgentResponse.success(objectMapper.writeValueAsString(conv));
        } catch (Exception e) {
            log.error("序列化会话失败", e);
            return AgentResponse.error("会话不存在");
        }
    }

    /**
     * 删除单个会话
     */
    @DeleteMapping("/conversation/{id}")
    public AgentResponse deleteConversation(@PathVariable Long id) {
        ConversationEntity conv = conversationService.getConversation(id);
        if (conv == null) {
            return AgentResponse.error("会话不存在");
        }
        conversationService.deleteConversation(id);
        return AgentResponse.success("会话已删除");
    }

    /**
     * 清空用户所有会话
     */
    @DeleteMapping("/conversations")
    public AgentResponse clearConversations(@RequestParam String userId) {
        conversationService.clearUserConversations(userId);
        return AgentResponse.success("用户会话已清空");
    }

    /**
     * 获取可用工具列表
     */
    @GetMapping("/tools")
    public List<Map<String, String>> getTools() {
        List<Map<String, String>> toolList = new ArrayList<>();
        for (Tool tool : toolRegistry.getAll()) {
            Map<String, String> toolInfo = new HashMap<>();
            toolInfo.put("name", tool.getName());
            toolInfo.put("description", tool.getDescription());
            toolList.add(toolInfo);
        }
        return toolList;
    }

    /**
     * 健康检查
     */
    @GetMapping("/health")
    public Map<String, Object> health() {
        Map<String, Object> status = new HashMap<>();
        status.put("status", "UP");
        status.put("model", chatModel.getModelName());
        status.put("tools", toolRegistry.getAll().size());
        return status;
    }

    // ============ 私有方法 ============

    private String getUserId(AgentRequest request) {
        if (request.getContext() != null && request.getContext().get("userId") != null) {
            return String.valueOf(request.getContext().get("userId"));
        }
        return "default";
    }

    private List<Memory.Message> loadMessages(ConversationEntity conversation) {
        List<Memory.Message> messages = new ArrayList<>();
        try {
            if (conversation.getMessages() != null && !conversation.getMessages().isEmpty()) {
                // 反序列化消息历史
                List<Map<String, Object>> msgList = objectMapper.readValue(
                    conversation.getMessages(),
                    objectMapper.getTypeFactory().constructCollectionType(List.class, Map.class)
                );
                for (Map<String, Object> msgData : msgList) {
                    String roleStr = (String) msgData.get("role");
                    String content = (String) msgData.get("content");
                    Memory.Message.Role role = Memory.Message.Role.valueOf(roleStr.toUpperCase());
                    messages.add(new Memory.Message(role, content));
                }
            }
        } catch (Exception e) {
            log.warn("加载消息历史失败，使用空列表", e);
        }

        // 添加系统提示（如果还没有）
        if (messages.isEmpty() || messages.get(0).role() != Memory.Message.Role.SYSTEM) {
            String systemPrompt = conversation.getSystemPrompt();
            if (systemPrompt == null || systemPrompt.isEmpty()) {
                systemPrompt = buildDefaultSystemPrompt();
            }
            messages.add(0, new Memory.Message(Memory.Message.Role.SYSTEM, systemPrompt));
        }

        return messages;
    }

    private void saveMessages(Long conversationId, List<Memory.Message> messages) {
        try {
            List<Map<String, String>> msgList = new ArrayList<>();
            for (Memory.Message msg : messages) {
                if (msg.role() != Memory.Message.Role.SYSTEM) {
                    Map<String, String> msgData = new HashMap<>();
                    msgData.put("role", msg.role().name().toLowerCase());
                    msgData.put("content", String.valueOf(msg.content()));
                    msgList.add(msgData);
                }
            }
            String messagesJson = objectMapper.writeValueAsString(msgList);
            conversationService.updateMessages(conversationId, messagesJson);
        } catch (Exception e) {
            // Propagate so chat() can decide how to surface it (e.g. metrics + log)
            // without silently losing the conversation history.
            throw new RuntimeException("保存会话历史失败", e);
        }
    }

    private ChatLoopResult executeReActLoop(List<Memory.Message> messages, List<Tool> activeTools, int maxIterations) {
        String finalAnswer = null;
        boolean timedOut = false;
        int iterations = 0;
        Set<String> toolsUsed = new LinkedHashSet<>();

        for (int step = 0; step < maxIterations; step++) {
            iterations = step + 1;
            ModelResponse response = chatModel.chat(messages);
            String content = response.getContent();

            if (content == null || content.isEmpty()) {
                return new ChatLoopResult("模型返回为空", false, iterations, toolsUsed);
            }

            if (response.isDone()) {
                finalAnswer = extractFinalAnswer(content);
                break;
            }

            // 解析工具调用
            List<ModelResponse.ToolCall> toolCalls = parseToolCalls(content);
            if (toolCalls.isEmpty()) {
                finalAnswer = content;
                break;
            }

            // 执行工具调用（仅限激活的工具）
            for (ModelResponse.ToolCall tc : toolCalls) {
                Tool tool = findTool(activeTools, tc.getName());
                toolsUsed.add(tc.getName());
                String toolResult;
                if (tool != null) {
                    Tool.ToolResult result = tool.execute(tc.getArguments());
                    toolResult = String.format("[TOOL: %s] %s",
                        tc.getName(),
                        result.success() ? result.output() : "Error: " + result.error());
                } else {
                    toolResult = "[TOOL: " + tc.getName() + "] Tool not available";
                }
                messages.add(new Memory.Message(Memory.Message.Role.ASSISTANT, content));
                messages.add(new Memory.Message(Memory.Message.Role.TOOL, toolResult, tc.getName()));
            }
        }

        if (finalAnswer == null) {
            finalAnswer = "达到最大迭代次数限制";
            timedOut = true;
        }

        return new ChatLoopResult(finalAnswer, timedOut, iterations, toolsUsed);
    }

    private Tool findTool(List<Tool> tools, String name) {
        for (Tool t : tools) {
            if (t.getName().equals(name)) return t;
        }
        return null;
    }

    /** ReAct 循环执行结果 */
    private record ChatLoopResult(String answer, boolean timedOut, int iterations, Set<String> toolsUsed) {}

    private String buildDefaultSystemPrompt() {
        return """
            你是一个智能助手，使用ReAct推理模式。

            推理规则：
            1. Thought: 思考需要做什么
            2. Action: 如果需要，调用工具（格式：Action: 工具名, Action Input: {"param": "value"}）
            3. Observation: 观察结果
            4. 重复直到得到最终答案
            5. Final Answer: 最终答案

            可用工具：
            - search: 搜索工具，用于查询信息
            - calculator: 计算器，用于数学计算
            - datetime: 日期时间工具，获取当前日期时间
            - ocr: OCR文字识别工具，从图片中提取文字

            当你确定知道答案时，直接返回：Final Answer: 你的答案
            """;
    }

    private List<ModelResponse.ToolCall> parseToolCalls(String content) {
        List<ModelResponse.ToolCall> calls = new ArrayList<>();

        if (content.contains("Action:") && content.contains("Action Input:")) {
            try {
                int actionIdx = content.indexOf("Action:");
                int inputIdx = content.indexOf("Action Input:");

                String actionPart = content.substring(actionIdx + 7, inputIdx).trim();
                String toolName = actionPart.split("\n")[0].trim();

                String inputPart = content.substring(inputIdx + 13).trim();
                int jsonStart = inputPart.indexOf("{");
                int jsonEnd = inputPart.lastIndexOf("}");

                if (jsonStart >= 0 && jsonEnd > jsonStart) {
                    String json = inputPart.substring(jsonStart, jsonEnd + 1);
                    Map<String, Object> args = new HashMap<>();
                    json = json.substring(1, json.length() - 1);
                    for (String pair : json.split(",")) {
                        String[] kv = pair.split(":");
                        if (kv.length == 2) {
                            String key = kv[0].trim().replace("\"", "");
                            String value = kv[1].trim().replace("\"", "");
                            args.put(key, value);
                        }
                    }
                    calls.add(new ModelResponse.ToolCall(UUID.randomUUID().toString(), toolName, args));
                }
            } catch (Exception e) {
                // Parsing failed — surface as warn so silent tool-call drops show up in logs.
                // The model may have produced a valid tool call we couldn't parse; the chat
                // response still returns the raw content so the caller is not lied to.
                log.warn("解析工具调用失败，已忽略本次 Action（model 输出: {}）",
                        content.length() > 200 ? content.substring(0, 200) + "..." : content, e);
            }
        }

        return calls;
    }

    private String extractFinalAnswer(String content) {
        if (content.contains("Final Answer:")) {
            int idx = content.indexOf("Final Answer:");
            return content.substring(idx + 13).trim();
        }
        return content;
    }

    private String extractPureBase64(String base64Data) {
        if (base64Data.contains(",")) {
            return base64Data.split(",")[1];
        }
        return base64Data;
    }

    private String extractImageFormat(String fileName) {
        if (fileName == null) return "jpeg";
        String lower = fileName.toLowerCase();
        if (lower.endsWith(".png")) return "png";
        if (lower.endsWith(".webp")) return "webp";
        if (lower.endsWith(".gif")) return "gif";
        return "jpeg";
    }
}
