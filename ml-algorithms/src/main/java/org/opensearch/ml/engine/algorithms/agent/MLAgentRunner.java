package org.opensearch.ml.engine.algorithms.agent;

import org.opensearch.core.action.ActionListener;
import org.opensearch.ml.common.agent.MLAgent;

import java.util.Map;

public interface MLAgentRunner {

    void run(MLAgent mlAgent, Map<String, String> params, ActionListener<Object> listener);
}
