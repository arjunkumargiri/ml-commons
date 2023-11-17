/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.tools;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

import java.time.Instant;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;
import org.opensearch.action.ActionType;
import org.opensearch.client.AdminClient;
import org.opensearch.client.ClusterAdminClient;
import org.opensearch.client.IndicesAdminClient;
import org.opensearch.client.node.NodeClient;
import org.opensearch.commons.alerting.action.GetAlertsResponse;
import org.opensearch.commons.alerting.model.Alert;
import org.opensearch.core.action.ActionListener;
import org.opensearch.ml.common.spi.tools.Tool;

public class SearchAlertsToolTests {
    @Mock
    private NodeClient nodeClient;
    @Mock
    private AdminClient adminClient;
    @Mock
    private IndicesAdminClient indicesAdminClient;
    @Mock
    private ClusterAdminClient clusterAdminClient;

    private Tool tool;

    private Map<String, String> nullParams;
    private Map<String, String> emptyParams;
    private Map<String, String> nonEmptyParams;

    @Before
    public void setup() {
        MockitoAnnotations.openMocks(this);
        tool = new SearchAlertsTool(nodeClient);

        nullParams = null;
        emptyParams = Collections.emptyMap();
        nonEmptyParams = Map.of("searchString", "foo");
    }

    @Test
    public void testRunWithNoAlerts() throws Exception {
        GetAlertsResponse getAlertsResponse = new GetAlertsResponse(Collections.emptyList(), 0);
        String expectedResponseStr = "Alerts=[]TotalAlerts=0";

        @SuppressWarnings("unchecked")
        ActionListener<String> listener = Mockito.mock(ActionListener.class);

        doAnswer((invocation) -> {
            ActionListener<GetAlertsResponse> responseListener = invocation.getArgument(2);
            responseListener.onResponse(getAlertsResponse);
            return null;
        }).when(nodeClient).execute(any(ActionType.class), any(), any());

        tool.run(nonEmptyParams, nonEmptyParams, listener);
        ArgumentCaptor<String> responseCaptor = ArgumentCaptor.forClass(String.class);
        verify(listener, times(1)).onResponse(responseCaptor.capture());
        assertEquals(expectedResponseStr, responseCaptor.getValue());
    }

    @Test
    public void testRunWithAlerts() throws Exception {
        Alert alert1 = new Alert(
            "alert-id-1",
            1234,
            1,
            "monitor-id",
            "workflow-id",
            "workflow-name",
            "monitor-name",
            1234,
            null,
            "trigger-id",
            "trigger-name",
            Collections.emptyList(),
            Collections.emptyList(),
            Alert.State.ACKNOWLEDGED,
            Instant.now(),
            null,
            null,
            null,
            null,
            Collections.emptyList(),
            "test-severity",
            Collections.emptyList(),
            null,
            null,
            Collections.emptyList()
        );
        Alert alert2 = new Alert(
            "alert-id-2",
            1234,
            1,
            "monitor-id",
            "workflow-id",
            "workflow-name",
            "monitor-name",
            1234,
            null,
            "trigger-id",
            "trigger-name",
            Collections.emptyList(),
            Collections.emptyList(),
            Alert.State.ACKNOWLEDGED,
            Instant.now(),
            null,
            null,
            null,
            null,
            Collections.emptyList(),
            "test-severity",
            Collections.emptyList(),
            null,
            null,
            Collections.emptyList()
        );
        List<Alert> mockAlerts = List.of(alert1, alert2);

        GetAlertsResponse getAlertsResponse = new GetAlertsResponse(mockAlerts, mockAlerts.size());
        String expectedResponseStr = new StringBuilder()
            .append("Alerts=[")
            .append(alert1.toString())
            .append(alert2.toString())
            .append("]TotalAlerts=2")
            .toString();

        @SuppressWarnings("unchecked")
        ActionListener<String> listener = Mockito.mock(ActionListener.class);

        doAnswer((invocation) -> {
            ActionListener<GetAlertsResponse> responseListener = invocation.getArgument(2);
            responseListener.onResponse(getAlertsResponse);
            return null;
        }).when(nodeClient).execute(any(ActionType.class), any(), any());

        tool.run(nonEmptyParams, nonEmptyParams, listener);
        ArgumentCaptor<String> responseCaptor = ArgumentCaptor.forClass(String.class);
        verify(listener, times(1)).onResponse(responseCaptor.capture());
        assertEquals(expectedResponseStr, responseCaptor.getValue());
    }

    @Test
    public void testValidate() {
        assertEquals(SearchAlertsTool.TYPE, tool.getType());
        assertTrue(tool.validate(emptyParams, emptyParams));
        assertTrue(tool.validate(nonEmptyParams, nonEmptyParams));
        assertTrue(tool.validate(nullParams, nullParams));
    }
}
