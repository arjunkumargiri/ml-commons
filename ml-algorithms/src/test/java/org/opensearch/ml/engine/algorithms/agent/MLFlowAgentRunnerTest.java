package org.opensearch.ml.engine.algorithms.agent;

import static org.mockito.Mockito.when;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.google.gson.Gson;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Captor;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;
import org.mockito.stubbing.Answer;
import org.opensearch.action.StepListener;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.ml.common.agent.MLAgent;
import org.opensearch.ml.common.agent.MLMemorySpec;
import org.opensearch.ml.common.agent.MLToolSpec;
import org.opensearch.ml.common.conversation.ActionConstants;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.common.spi.memory.Memory;
import org.opensearch.ml.common.spi.tools.Tool;
import org.opensearch.ml.engine.memory.ConversationIndexMemory;

import org.opensearch.ml.engine.memory.MLMemoryManager;
import software.amazon.awssdk.utils.ImmutableMap;

public class MLFlowAgentRunnerTest {

    public static final String FIRST_TOOL = "firstTool";
    public static final String SECOND_TOOL = "secondTool";

    @Mock
    private Client client;

    private Settings settings;

    @Mock
    private ClusterService clusterService;

    @Mock
    private NamedXContentRegistry xContentRegistry;

    private Map<String, Tool.Factory> toolFactories;

    private Map<String, Memory.Factory> memoryMap;

    private MLFlowAgentRunner mlFlowAgentRunner;

    @Mock
    private Tool.Factory firstToolFactory;

    @Mock
    private Tool.Factory secondToolFactory;
    @Mock
    private Tool firstTool;

    @Mock
    private Tool secondTool;

    @Mock
    private ConversationIndexMemory memory;

    @Mock
    private MLMemoryManager memoryManager;

    @Mock
    private ConversationIndexMemory.Factory mockMemoryFactory;

    @Mock
    private ActionListener<Object> agentActionListener;

    @Captor
    private ArgumentCaptor<Object> objectCaptor;

    @Captor
    private ArgumentCaptor<StepListener<Object>> nextStepListenerCaptor;
    
    @Captor
    private ArgumentCaptor<Map<String, String>> toolParamsCaptor;

    @Before
    @SuppressWarnings("unchecked")
    public void setup() {
        MockitoAnnotations.openMocks(this);
        settings = Settings.builder().build();
        toolFactories = ImmutableMap.of(FIRST_TOOL, firstToolFactory, SECOND_TOOL, secondToolFactory);
        memoryMap = ImmutableMap.of("memoryType", mockMemoryFactory);
        mlFlowAgentRunner = new MLFlowAgentRunner(client, settings, clusterService, xContentRegistry, toolFactories, memoryMap);
        when(firstToolFactory.create(Mockito.anyMap())).thenReturn(firstTool);
        when(secondToolFactory.create(Mockito.anyMap())).thenReturn(secondTool);
        Mockito
            .doAnswer(generateToolResponse("First tool response"))
            .when(firstTool)
            .run(Mockito.anyMap(), nextStepListenerCaptor.capture());
        Mockito
            .doAnswer(generateToolResponse("Second tool response"))
            .when(secondTool)
            .run(Mockito.anyMap(), nextStepListenerCaptor.capture());
    }

    private Answer generateToolResponse(Object response) {
        return invocation -> {
            ActionListener<Object> listener = invocation.getArgument(1);
            listener.onResponse(response);
            return null;
        };
    }

    @Test
    public void test_HappyCase_GenerateSuccessfulResponse() {
        final MLAgent mlAgent = getMlAgent(getMLToolSpec(FIRST_TOOL), getMLToolSpec(SECOND_TOOL));
        mlFlowAgentRunner.run(mlAgent, getParams(), agentActionListener);
        Mockito.verify(agentActionListener).onResponse(objectCaptor.capture());

        Mockito.verify(firstTool).run(Mockito.anyMap(), Mockito.any());
        Mockito.verify(secondTool).run(Mockito.anyMap(), Mockito.any());
        List<ModelTensor> agentOutput = (List<ModelTensor>) objectCaptor.getValue();
        Assert.assertEquals(1, agentOutput.size());
        // Respond with last tool output
        Assert.assertEquals(SECOND_TOOL, agentOutput.get(0).getName());
        Assert.assertEquals("Second tool response", agentOutput.get(0).getResult());
    }


    @Test
    public void test_SingleTool_GenerateSuccessfulResponse() {
        final MLAgent mlAgent = getMlAgent(getMLToolSpec(FIRST_TOOL));
        mlFlowAgentRunner.run(mlAgent, getParams(), agentActionListener);

        Mockito.verify(agentActionListener).onResponse(objectCaptor.capture());
        Mockito.verify(firstTool).run(Mockito.anyMap(), Mockito.any());
        List<ModelTensor> agentOutput = (List<ModelTensor>) objectCaptor.getValue();
        Assert.assertEquals(1, agentOutput.size());
        // Respond with last tool output
        Assert.assertEquals(FIRST_TOOL, agentOutput.get(0).getName());
        Assert.assertEquals("First tool response", agentOutput.get(0).getResult());
    }

    @Test
    public void test_ZeroTool_ThrowsValidationError() {
        final MLAgent mlAgent = getMlAgent();
        mlFlowAgentRunner.run(mlAgent, getParams(), agentActionListener);
        Mockito.verify(agentActionListener).onFailure(Mockito.any());
    }

    @Test
    public void testRunWithIncludeOutputNotSet() {
        final MLAgent mlAgent = getMlAgent(getMLToolSpec(FIRST_TOOL), getMLToolSpec(SECOND_TOOL));
        mlFlowAgentRunner.run(mlAgent, getParams(), agentActionListener);
        Mockito.verify(agentActionListener).onResponse(objectCaptor.capture());
        List<ModelTensor> agentOutput = (List<ModelTensor>) objectCaptor.getValue();
        Assert.assertEquals(1, agentOutput.size());
        // Respond with last tool output
        Assert.assertEquals(SECOND_TOOL, agentOutput.get(0).getName());
        Assert.assertEquals("Second tool response", agentOutput.get(0).getResult());
    }

    @Test
    public void testRunWithIncludeOutputSet() {
        MLToolSpec firstToolSpec = MLToolSpec.builder().name(FIRST_TOOL).type(FIRST_TOOL).includeOutputInAgentResponse(true).build();
        MLToolSpec secondToolSpec = MLToolSpec.builder().name(SECOND_TOOL).type(SECOND_TOOL).includeOutputInAgentResponse(true).build();
        final MLAgent mlAgent = getMlAgent(firstToolSpec, secondToolSpec);
        mlFlowAgentRunner.run(mlAgent, getParams(), agentActionListener);
        Mockito.verify(agentActionListener).onResponse(objectCaptor.capture());
        List<ModelTensor> agentOutput = (List<ModelTensor>) objectCaptor.getValue();
        // Respond with all tool output
        Assert.assertEquals(2, agentOutput.size());
        Assert.assertEquals(FIRST_TOOL, agentOutput.get(0).getName());
        Assert.assertEquals(SECOND_TOOL, agentOutput.get(1).getName());
        Assert.assertEquals("First tool response", agentOutput.get(0).getResult());
        Assert.assertEquals("Second tool response", agentOutput.get(1).getResult());
    }

    @Test
    public void testWithMemoryNotSet() {
        MLToolSpec firstToolSpec = MLToolSpec.builder().name(FIRST_TOOL).type(FIRST_TOOL).build();
        MLToolSpec secondToolSpec = MLToolSpec.builder().name(SECOND_TOOL).type(SECOND_TOOL).build();
        final MLAgent mlAgent = MLAgent
            .builder()
            .name("TestAgent")
            .memory(null)
            .tools(Arrays.asList(firstToolSpec, secondToolSpec))
            .build();
        mlFlowAgentRunner.run(mlAgent, getParams(), agentActionListener);
        Mockito.verify(agentActionListener).onResponse(objectCaptor.capture());
        List<ModelTensor> agentOutput = (List<ModelTensor>) objectCaptor.getValue();
        Assert.assertEquals(1, agentOutput.size());
        // Respond with last tool output
        Assert.assertEquals(SECOND_TOOL, agentOutput.get(0).getName());
        Assert.assertEquals("Second tool response", agentOutput.get(0).getResult());
    }

    @Test
    public void test_MemorySet_UpdateInteraction() {
        final Map<String, String> params = getParams();
        params.put(MLAgentExecutor.PARENT_INTERACTION_ID, "test_interaction_id");
        Mockito.doAnswer(invocation -> {
            ActionListener<Memory> listener = invocation.getArgument(1);
            listener.onResponse(memory);
            return null;
        }).when(mockMemoryFactory).create(Mockito.eq("memoryId"), Mockito.any(ActionListener.class));
        Mockito.when(memory.getMemoryManager()).thenReturn(memoryManager);
        final MLAgent mlAgent = getMlAgent(getMLToolSpec(FIRST_TOOL));
        mlFlowAgentRunner.run(mlAgent, params, agentActionListener);

        Mockito.verify(memory).getMemoryManager();
        Map<String, Object> expectedInteraction = ImmutableMap.of(ActionConstants.ADDITIONAL_INFO_FIELD,
                ImmutableMap.of("firstTool.output", "First tool response"));
        Mockito.verify(memoryManager).updateInteraction(
                Mockito.eq("test_interaction_id"), Mockito.eq(expectedInteraction), Mockito.any());
    }
    
    @Test
    public void test_ToolParams_ValidateToolSpecParamsIsIncluded() {
        final Map<String, String> params = getParams();
        params.put("test_param_key", "test_param_value");
        MLToolSpec firstToolSpec = MLToolSpec.builder().name(FIRST_TOOL).type(FIRST_TOOL)
                .parameters(ImmutableMap.of("test_toolspec_key", "test_toolspec_value")).build();
        final MLAgent mlAgent = getMlAgent(firstToolSpec);
        mlFlowAgentRunner.run(mlAgent, params, agentActionListener);
        Mockito.verify(firstTool).run(toolParamsCaptor.capture(), Mockito.any());
        Assert.assertEquals("test_param_value", toolParamsCaptor.getValue().get("test_param_key"));
        Assert.assertEquals("test_toolspec_value", toolParamsCaptor.getValue().get("test_toolspec_key"));
    }

    @Test
    public void test_ToolParams_ValidateKeywordReplaced() {
        final Map<String, String> params = getParams();
        params.put("test_param_key", "test_param_value");
        params.put(FIRST_TOOL + ".test_param", "test_input_value");
        MLToolSpec firstToolSpec = MLToolSpec.builder().name(FIRST_TOOL).type(FIRST_TOOL).build();
        final MLAgent mlAgent = getMlAgent(firstToolSpec);
        mlFlowAgentRunner.run(mlAgent, params, agentActionListener);
        Mockito.verify(firstTool).run(toolParamsCaptor.capture(), Mockito.any());
        Assert.assertEquals("test_param_value", toolParamsCaptor.getValue().get("test_param_key"));
        Assert.assertEquals("test_input_value", toolParamsCaptor.getValue().get("test_param"));
    }

    @Test
    public void test_ToolParams_ValidateInputParamsReplaced() {
        final Map<String, String> params = getParams();
        params.put("test_param_key", "test_param_value");
        params.put("input", "Check if value is replaced: ${parameters.test_param_key}");
        MLToolSpec firstToolSpec = MLToolSpec.builder().name(FIRST_TOOL).type(FIRST_TOOL).build();
        final MLAgent mlAgent = getMlAgent(firstToolSpec);
        mlFlowAgentRunner.run(mlAgent, params, agentActionListener);
        Mockito.verify(firstTool).run(toolParamsCaptor.capture(), Mockito.any());
        Assert.assertEquals("test_param_value", toolParamsCaptor.getValue().get("test_param_key"));
        Assert.assertEquals("Check if value is replaced: test_param_value", toolParamsCaptor.getValue().get("input"));
    }

    @Test(expected = IllegalArgumentException.class)
    public void test_CreateTool_InvalidTool_ThrowsException() {
        final MLAgent mlAgent = getMlAgent(getMLToolSpec("invalid_tool"));
        mlFlowAgentRunner.run(mlAgent, getParams(), agentActionListener);
    }

    @Test
    public void test_HandleToolResponse_ModelTensorOutput() {
        ModelTensor modelTensor = ModelTensor.builder().name(FIRST_TOOL).result("First tool response").build();
        ModelTensors modelTensors = ModelTensors.builder().mlModelTensors(Arrays.asList(modelTensor)).build();
        ModelTensorOutput modelTensorOutput = new ModelTensorOutput(Arrays.asList(modelTensors));
        Mockito
                .doAnswer(generateToolResponse(modelTensorOutput))
                .when(firstTool)
                .run(Mockito.anyMap(), nextStepListenerCaptor.capture());
        final MLAgent mlAgent = getMlAgent(getMLToolSpec(FIRST_TOOL));
        mlFlowAgentRunner.run(mlAgent, getParams(), agentActionListener);

        Mockito.verify(agentActionListener).onResponse(objectCaptor.capture());
        Mockito.verify(firstTool).run(Mockito.anyMap(), Mockito.any());
        List<ModelTensor> agentOutput = (List<ModelTensor>) objectCaptor.getValue();
        Assert.assertEquals(1, agentOutput.size());
        // Respond with last tool output
        Assert.assertEquals(FIRST_TOOL, agentOutput.get(0).getName());
        Assert.assertEquals("First tool response", agentOutput.get(0).getResult());
    }

    @Test
    public void test_HandleToolResponse_ModelTensor() {
        ModelTensor modelTensor = ModelTensor.builder().name(FIRST_TOOL).result("First tool response").build();
        Mockito
                .doAnswer(generateToolResponse(modelTensor))
                .when(firstTool)
                .run(Mockito.anyMap(), nextStepListenerCaptor.capture());
        final MLAgent mlAgent = getMlAgent(getMLToolSpec(FIRST_TOOL));
        mlFlowAgentRunner.run(mlAgent, getParams(), agentActionListener);

        Mockito.verify(agentActionListener).onResponse(objectCaptor.capture());
        Mockito.verify(firstTool).run(Mockito.anyMap(), Mockito.any());
        List<ModelTensor> agentOutput = (List<ModelTensor>) objectCaptor.getValue();
        Assert.assertEquals(1, agentOutput.size());
        // Respond with last tool output
        Assert.assertEquals(FIRST_TOOL, agentOutput.get(0).getName());
        Assert.assertEquals(new Gson().toJson(modelTensor), agentOutput.get(0).getResult());
    }

    @Test
    public void test_HandleToolResponse_ModelTensors() {
        ModelTensor modelTensor = ModelTensor.builder().name(FIRST_TOOL).result("First tool response").build();
        ModelTensors modelTensors = ModelTensors.builder().mlModelTensors(Arrays.asList(modelTensor)).build();
        Mockito
                .doAnswer(generateToolResponse(modelTensors))
                .when(firstTool)
                .run(Mockito.anyMap(), nextStepListenerCaptor.capture());
        final MLAgent mlAgent = getMlAgent(getMLToolSpec(FIRST_TOOL));
        mlFlowAgentRunner.run(mlAgent, getParams(), agentActionListener);

        Mockito.verify(agentActionListener).onResponse(objectCaptor.capture());
        Mockito.verify(firstTool).run(Mockito.anyMap(), Mockito.any());
        List<ModelTensor> agentOutput = (List<ModelTensor>) objectCaptor.getValue();
        Assert.assertEquals(1, agentOutput.size());
        // Respond with last tool output
        Assert.assertEquals(FIRST_TOOL, agentOutput.get(0).getName());
        Assert.assertEquals(new Gson().toJson(modelTensors), agentOutput.get(0).getResult());
    }

    private Map<String, String> getParams() {
        final Map<String, String> params = new HashMap<>();
        params.put(MLAgentExecutor.MEMORY_ID, "memoryId");
        return params;
    }

    private MLToolSpec getMLToolSpec(String toolName) {
        return MLToolSpec.builder().name(toolName).type(toolName).build();
    }

    private MLAgent getMlAgent(MLToolSpec ...tools) {
        MLMemorySpec mlMemorySpec = MLMemorySpec.builder().type("memoryType").build();
        return MLAgent
                .builder()
                .name("TestAgent")
                .memory(mlMemorySpec)
                .tools(Arrays.asList(tools))
                .build();
    }

}
