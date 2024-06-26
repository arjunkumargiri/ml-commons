package org.opensearch.ml.dao.connector;

import lombok.extern.log4j.Log4j2;
import org.opensearch.OpenSearchStatusException;
import org.opensearch.action.get.GetRequest;
import org.opensearch.action.get.GetResponse;
import org.opensearch.action.index.IndexRequest;
import org.opensearch.action.index.IndexResponse;
import org.opensearch.action.support.WriteRequest;
import org.opensearch.client.Client;
import org.opensearch.client.Response;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.IndexNotFoundException;
import org.opensearch.ml.common.connector.Connector;
import org.opensearch.ml.common.transport.connector.MLConnectorGetResponse;
import org.opensearch.ml.common.transport.connector.MLCreateConnectorResponse;
import org.opensearch.ml.engine.indices.MLIndicesHandler;
import org.opensearch.search.fetch.subphase.FetchSourceContext;

import java.io.IOException;
import java.util.Optional;

import static org.opensearch.core.xcontent.XContentParserUtils.ensureExpectedToken;
import static org.opensearch.ml.common.CommonValue.ML_CONNECTOR_INDEX;
import static org.opensearch.ml.utils.MLNodeUtils.createXContentParserFromRegistry;
import static org.opensearch.ml.utils.RestActionUtils.getFetchSourceContext;

@Log4j2
public class OpenSearchTransportConnectorDao implements ConnectorDao {

    private Client client;
    private MLIndicesHandler mlIndicesHandler;

    private NamedXContentRegistry xContentRegistry;

    public OpenSearchTransportConnectorDao(Client client,
                                           MLIndicesHandler mlIndicesHandler,
                                           NamedXContentRegistry xContentRegistry) {
        this.client = client;
        this.mlIndicesHandler = mlIndicesHandler;
        this.xContentRegistry = xContentRegistry;
    }

    @Override
    public String createConnector(Connector connector) throws Exception {
        mlIndicesHandler.initMLConnectorIndex();
        try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
            IndexRequest indexRequest = new IndexRequest(ML_CONNECTOR_INDEX);
            indexRequest.source(connector.toXContent(XContentBuilder.builder(XContentType.JSON.xContent()), ToXContent.EMPTY_PARAMS));
            indexRequest.setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
            final IndexResponse indexResponse = client.index(indexRequest).actionGet();
            context.restore();
            return indexResponse.getId();
        }
    }

    @Override
    public Optional<Connector> getConnector(String connectorId, boolean isReturnContent) throws Exception {
        FetchSourceContext fetchSourceContext = getFetchSourceContext(isReturnContent);
        GetRequest getRequest = new GetRequest(ML_CONNECTOR_INDEX).id(connectorId).fetchSourceContext(fetchSourceContext);

        ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext();
        try {
            GetResponse r = client.get(getRequest).actionGet();
            log.debug("Completed Get Connector Request, id:{}", connectorId);

            if (r != null && r.isExists()) {
                try (XContentParser parser = createXContentParserFromRegistry(xContentRegistry, r.getSourceAsBytesRef())) {
                    ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.nextToken(), parser);
                    Connector mlConnector = Connector.createConnector(parser);
                    return Optional.of(mlConnector);
                } catch (Exception e) {
                    log.error("Failed to parse ml connector" + r.getId(), e);
                    throw e;
                }
            }
            return Optional.empty();
        } catch(Exception e) {
            if (e instanceof IndexNotFoundException) {
                log.error("Failed to get connector index", e);
                throw new OpenSearchStatusException("Failed to find connector", RestStatus.NOT_FOUND);
            } else {
                log.error("Failed to get ML connector " + connectorId, e);
                throw e;
            }
        } finally {
            context.restore();
        }

    }
}
