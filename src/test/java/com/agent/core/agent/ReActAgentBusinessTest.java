package com.agent.core.agent;

import com.agent.core.memory.Memory;
import com.agent.core.model.ChatModel;
import com.agent.core.model.ModelResponse;
import com.agent.core.tool.Tool;
import com.agent.core.tool.ToolRegistry;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Business-logic tests for ReActAgent's loop bookkeeping that the pre-existing
 * tests do not exercise: input validation, tool-failure handling, max-step
 * result enrichment, and safe encoding of tool output containing
 * quotes/newlines/carriage-returns.
 *
 * The test uses an inline stub ChatModel so the suite is fully deterministic
 * and network-free.
 */
@DisplayName("ReActAgent loop edge cases and bookkeeping")
class ReActAgentBusinessTest {

    private TestChatModel chatModel;
    private ReActAgent agent;

    @BeforeEach
    void setUp() {
        chatModel = new TestChatModel();
        agent = new ReActAgent("test-agent", "Test Agent", "you are a test agent", new ToolRegistry(), 5, chatModel);
        agent.addTool(new EchoTool("echo"));
    }

    // ==================== Input validation ====================

    @Test
    @DisplayName("null input is rejected without a model call")
    void nullInputRejected() {
        ModelResponse r = agent.execute((String) null, Map.of());
        assertTrue(r.isDone());
        assertNotNull(r.getContent());
        String c = r.getContent().toLowerCase();
        assertTrue(c.contains("input") || c.contains("输入"),
                "error response should mention input/输入, got: " + r.getContent());
        assertEquals(0, chatModel.callCount.get(), "no model calls should be made for null input");
    }

    @Test
    @DisplayName("blank input is rejected without a model call")
    void blankInputRejected() {
        ModelResponse r = agent.execute("   \t  ", Map.of());
        assertTrue(r.isDone());
        assertEquals(0, chatModel.callCount.get());
    }

    @Test
    @DisplayName("empty input is rejected without a model call")
    void emptyInputRejected() {
        ModelResponse r = agent.execute("", Map.of());
        assertTrue(r.isDone());
        assertEquals(0, chatModel.callCount.get());
    }

    // ==================== Tool failure handling ====================

    @Test
    @DisplayName("tool failure result is recorded in history for the next model call")
    void toolFailureRecordedInHistory() {
        AtomicInteger callIdx = new AtomicInteger();
        chatModel.responseAt = i -> {
            if (i == 0) {
                return ModelResponse.builder()
                        .content("calling failing tool")
                        .toolCalls(new ModelResponse.ToolCall[]{
                                ModelResponse.ToolCall.builder()
                                        .id("call-1")
                                        .name("failing")
                                        .arguments(Map.of())
                                        .build()
                        })
                        .done(false)
                        .build();
            }
            return ModelResponse.builder()
                    .content("got failure")
                    .done(true)
                    .build();
        };
        agent.addTool(new Tool() {
            @Override public String getName() { return "failing"; }
            @Override public String getDescription() { return "always fails"; }
            @Override public String getParameterSchema() { return "{}"; }
            @Override public ToolResult execute(Map<String, Object> p) {
                return ToolResult.failure("simulated network error");
            }
        });

        ModelResponse r = agent.execute("do something", Map.of());

        assertTrue(r.isDone(), "agent should reach done after second model call");
        List<Memory.Message> lastCallMessages = chatModel.lastCallMessages;
        assertNotNull(lastCallMessages);
        boolean sawToolMessage = lastCallMessages.stream()
                .anyMatch(m -> m.role() == Memory.Message.Role.TOOL);
        assertTrue(sawToolMessage, "tool result message should be in history, got: " + lastCallMessages);

        // The TOOL message must record that it was a failure (not pretend success)
        String toolMsg = lastCallMessages.stream()
                .filter(m -> m.role() == Memory.Message.Role.TOOL)
                .map(m -> (String) m.content())
                .findFirst().orElseThrow();
        assertTrue(toolMsg.contains("false") || toolMsg.toLowerCase().contains("fail")
                        || toolMsg.contains("error") || toolMsg.contains("失败"),
                "tool message must indicate failure, got: " + toolMsg);
    }

    @Test
    @DisplayName("tool output containing quotes and newlines is preserved intact in history")
    void toolOutputSafelyEscaped() {
        chatModel.responseAt = i -> {
            if (i == 0) {
                return ModelResponse.builder()
                        .toolCalls(new ModelResponse.ToolCall[]{
                                ModelResponse.ToolCall.builder()
                                        .id("call-2")
                                        .name("noisy")
                                        .arguments(Map.of())
                                        .build()
                        })
                        .done(false)
                        .build();
            }
            return ModelResponse.builder().content("done").done(true).build();
        };
        agent.addTool(new Tool() {
            @Override public String getName() { return "noisy"; }
            @Override public String getDescription() { return ""; }
            @Override public String getParameterSchema() { return "{}"; }
            @Override public ToolResult execute(Map<String, Object> p) {
                return ToolResult.success("line1\nline2 \"quoted\" with\rcarriage");
            }
        });

        ModelResponse r = agent.execute("test", Map.of());
        assertTrue(r.isDone());

        List<Memory.Message> lastCallMessages = chatModel.lastCallMessages;
        String toolMsg = lastCallMessages.stream()
                .filter(m -> m.role() == Memory.Message.Role.TOOL)
                .map(m -> (String) m.content())
                .findFirst()
                .orElseThrow();
        assertTrue(toolMsg.contains("line1"), "tool message must contain original line1: " + toolMsg);
        assertTrue(toolMsg.contains("line2"), "tool message must contain original line2: " + toolMsg);
        // Quotes are JSON-escaped for safe transport; verify the escaped form is intact
        assertTrue(toolMsg.contains("\\\"" + "quoted" + "\\\""),
                "tool message must preserve quoted text (JSON-escaped): " + toolMsg);
        assertTrue(toolMsg.contains("carriage"), "tool message must contain original carriage: " + toolMsg);
    }

    // ==================== Max-step result enrichment ====================

    @Test
    @DisplayName("reaching maxSteps returns a done response that mentions the limit")
    void maxStepsReturnsDoneWithLimitMessage() {
        chatModel.responseAt = i -> ModelResponse.builder()
                .toolCalls(new ModelResponse.ToolCall[]{
                        ModelResponse.ToolCall.builder()
                                .id("call-3")
                                .name("nonexistent")
                                .arguments(Map.of())
                                .build()
                })
                .done(false)
                .build();
        ReActAgent limited = new ReActAgent("limited", "limited agent", "sys", new ToolRegistry(), 3, chatModel);
        ModelResponse r = limited.execute("loop", Map.of());
        assertTrue(r.isDone(), "should be done after hitting max steps");
        assertNotNull(r.getContent());
        String c = r.getContent().toLowerCase();
        assertTrue(c.contains("max") || c.contains("最大") || c.contains("step") || c.contains("步"),
                "max-step response should mention the limit, got: " + r.getContent());
    }

    // ==================== Tool registry wiring ====================

    @Test
    @DisplayName("removeTool removes a tool so it is no longer callable")
    void removeToolMakesItUnreachable() {
        agent.addTool(new EchoTool("temp"));
        assertTrue(agent.getTools().stream().anyMatch(t -> "temp".equals(t.getName())));
        agent.removeTool("temp");
        assertTrue(agent.getTools().stream().noneMatch(t -> "temp".equals(t.getName())));
    }

    @Test
    @DisplayName("addTool twice with the same name replaces, does not duplicate")
    void addToolReplaces() {
        agent.addTool(new EchoTool("dup"));
        agent.addTool(new EchoTool("dup"));
        long count = agent.getTools().stream().filter(t -> "dup".equals(t.getName())).count();
        assertEquals(1, count, "duplicate add should replace, not stack, got: " + agent.getTools().size());
    }

    // ==================== helpers ====================

    /** A ChatModel stub for testing. */
    static class TestChatModel implements ChatModel {
        volatile Function<Integer, ModelResponse> responseAt = i ->
                ModelResponse.builder().content("default").done(true).build();
        final AtomicInteger callCount = new AtomicInteger();
        volatile List<Memory.Message> lastCallMessages;

        @Override
        public ModelResponse chat(List<Memory.Message> messages) {
            return chat(messages, Map.of());
        }

        @Override
        public ModelResponse chat(List<Memory.Message> messages, Map<String, Object> parameters) {
            int idx = callCount.getAndIncrement();
            lastCallMessages = new ArrayList<>(messages);
            return responseAt.apply(idx);
        }

        @Override
        public String getModelName() {
            return "test-model";
        }
    }

    /** A no-op tool that always succeeds with its name as output. */
    static class EchoTool implements Tool {
        private final String name;
        EchoTool(String name) { this.name = name; }
        @Override public String getName() { return name; }
        @Override public String getDescription() { return "echoes name"; }
        @Override public String getParameterSchema() { return "{}"; }
        @Override public ToolResult execute(Map<String, Object> p) { return ToolResult.success("echo:" + name); }
    }
}
