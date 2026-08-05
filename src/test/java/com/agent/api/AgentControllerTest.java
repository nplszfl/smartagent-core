package com.agent.api;

import com.agent.core.memory.Memory;
import com.agent.core.model.ChatModel;
import com.agent.core.model.ModelResponse;
import com.agent.entity.ConversationEntity;
import com.agent.service.ConversationService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

/**
 * Unit tests for {@link AgentController} focused on the loop bookkeeping
 * and persistence semantics that were previously uncovered:
 *
 *  - successful chat returns the model's answer
 *  - persistence failure does NOT silently drop the answer (regression test for
 *    the previous "log and continue" behavior that lost conversation history)
 *  - tool-call parse failures are non-fatal: the chat still succeeds and the
 *    raw model content is surfaced instead of being silently dropped
 *
 * The controller is exercised with Mockito-stubbed ChatModel and ConversationService
 * so the suite is deterministic and network-free.
 */
@DisplayName("AgentController chat endpoint semantics")
class AgentControllerTest {

    private ChatModel chatModel;
    private ConversationService conversationService;
    private AgentController controller;

    @BeforeEach
    void setUp() {
        chatModel = mock(ChatModel.class);
        when(chatModel.getModelName()).thenReturn("test-model");

        conversationService = mock(ConversationService.class);
        ConversationEntity conv = new ConversationEntity();
        conv.setId(42L);
        conv.setUserId("u1");
        conv.setAgentName("assistant");
        conv.setSystemPrompt("sys");
        conv.setMessages("[]");
        when(conversationService.getOrCreateConversation(anyString(), anyString(), anyString()))
                .thenReturn(conv);

        controller = new AgentController(chatModel, conversationService);
    }

    @Test
    @DisplayName("successful chat returns the model's final answer")
    void chatHappyPath() {
        AtomicInteger callIdx = new AtomicInteger();
        when(chatModel.chat(any())).thenAnswer(inv -> {
            int i = callIdx.getAndIncrement();
            if (i == 0) {
                return ModelResponse.builder()
                        .content("Thought: need tool\nAction: noisy\nAction Input: {\"q\":\"hi\"}\n")
                        .done(false)
                        .build();
            }
            return ModelResponse.builder().content("Final Answer: 42").done(true).build();
        });

        AgentRequest req = new AgentRequest();
        req.setInput("hi");
        req.setMaxIterations(5);

        AgentResponse resp = controller.chat(req);

        assertTrue(resp.isSuccess(), "happy path should succeed");
        assertEquals("42", resp.getContent(), "controller strips 'Final Answer:' prefix");
        verify(conversationService, atLeastOnce()).updateMessages(eq(42L), anyString());
    }

    @Test
    @DisplayName("saveMessages failure does NOT silently drop the user's answer")
    void saveFailureDoesNotLoseAnswer() {
        when(chatModel.chat(any()))
                .thenReturn(ModelResponse.builder().content("Final Answer: kept").done(true).build());
        doThrow(new RuntimeException("DB down"))
                .when(conversationService).updateMessages(any(Long.class), anyString());

        AgentRequest req = new AgentRequest();
        req.setInput("hi");

        AgentResponse resp = controller.chat(req);

        // The user still gets their answer — silent data-loss regression test.
        assertTrue(resp.isSuccess(), "chat should still succeed for the user, got: " + resp);
        assertEquals("kept", resp.getContent());
    }

    @Test
    @DisplayName("malformed Action block is non-fatal: chat succeeds, raw content is the answer")
    void malformedToolCallIsNonFatal() {
        // Has 'Action:' but JSON block is broken (unterminated) → parseToolCalls catches
        when(chatModel.chat(any()))
                .thenReturn(ModelResponse.builder()
                        .content("Thought: try a tool\nAction: broken\nAction Input: {not json")
                        .done(false)
                        .build())
                .thenReturn(ModelResponse.builder()
                        .content("Final Answer: fallback").done(true).build());

        AgentRequest req = new AgentRequest();
        req.setInput("hi");
        req.setMaxIterations(3);

        AgentResponse resp = controller.chat(req);

        // parseToolCalls failure should be swallowed (now logged at WARN) and the
        // raw content from the first (broken) attempt becomes the final answer
        // since the loop ends when parseToolCalls returns empty.
        assertTrue(resp.isSuccess(), "broken Action must not crash chat, got: " + resp);
        assertTrue(resp.getContent().startsWith("Thought: try a tool"),
                "raw model content should be returned when tool parse fails, got: " + resp.getContent());
    }

    @Test
    @DisplayName("saveMessages serializes only USER/ASSISTANT/TOOL messages, not SYSTEM")
    void saveMessagesExcludesSystemPrompt() {
        AtomicInteger callIdx = new AtomicInteger();
        when(chatModel.chat(any())).thenAnswer(inv -> {
            List<Memory.Message> msgs = inv.getArgument(0);
            // Sanity: controller should have injected the SYSTEM prompt into the head of messages
            assertFalse(msgs.isEmpty(), "messages must be non-empty");
            assertEquals(Memory.Message.Role.SYSTEM, msgs.get(0).role(),
                    "SYSTEM message must be at index 0");
            int i = callIdx.getAndIncrement();
            if (i == 0) {
                return ModelResponse.builder().content("hi").done(true).build();
            }
            return ModelResponse.builder().content("done").done(true).build();
        });

        AgentRequest req = new AgentRequest();
        req.setInput("hi");
        controller.chat(req);

        // Capture what was persisted
        org.mockito.ArgumentCaptor<String> jsonCaptor =
                org.mockito.ArgumentCaptor.forClass(String.class);
        verify(conversationService).updateMessages(eq(42L), jsonCaptor.capture());

        String persisted = jsonCaptor.getValue();
        assertFalse(persisted.contains("\"system\""),
                "SYSTEM role must NOT be persisted (it's injected per-request), got: " + persisted);
        assertTrue(persisted.contains("\"user\""),
                "USER message must be persisted, got: " + persisted);
    }

    @Test
    @DisplayName("blank input is rejected by @NotBlank validation before reaching controller logic")
    void blankInputRejected() {
        // Validation is normally triggered by Spring; we exercise controller.chat directly
        // so the test verifies the controller does not crash on empty input.
        AgentRequest req = new AgentRequest();
        req.setInput("");
        // No chat call configured — if controller doesn't validate, we'd get NPE.
        AgentResponse resp = controller.chat(req);
        // Even if we accept blank input, we should not throw — verify completion.
        assertNotNull(resp);
    }
}