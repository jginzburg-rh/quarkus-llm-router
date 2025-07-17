package com.redhat.composer.services;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.redhat.composer.components.aiservices.AiServicesFactory;
import com.redhat.composer.components.aiservices.BaseAiService;
import com.redhat.composer.components.servingruntime.streaming.StreamingServingRuntimeFactory;
import com.redhat.composer.model.mongo.AssistantEntity;
import com.redhat.composer.model.mongo.LlmConnectionEntity;
import com.redhat.composer.model.mongo.RetrieverConnectionEntity;
import com.redhat.composer.model.request.AssistantChatRequest;
import com.redhat.composer.model.request.ChatBotRequest;
import com.redhat.composer.model.request.RetrieverRequest;
import com.redhat.composer.model.response.ContentResponse;
import com.redhat.composer.repositories.AssistantRepository;
import com.redhat.composer.util.mappers.MapperUtil;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.model.chat.StreamingChatLanguageModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import io.opentelemetry.api.trace.Span;
import io.quarkus.logging.Log;
import io.quarkus.runtime.util.StringUtil;
import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * ChatBotService class.
 */
@ApplicationScoped
public class ChatBotService {

  @ConfigProperty(name = "prompt.default.system.message")
  String defaultSystemMessage;

  @Inject
  StreamingServingRuntimeFactory modelTemplateFactory;

  @Inject
  AiServicesFactory aiServicesFactory;

  @Inject
  RetrieveService ragService;

  @Inject
  AssistantRepository assistantRepository;

  @Inject
  MapperUtil mapperUtil;

  @Inject
  ObjectMapper objectMapper;

  /**
   * Chat with an assistant.
   * @param request the AssistantChatRequest
   * @return stream of chat response
   */
  public Multi<String> chat(AssistantChatRequest request) {
    return chat(request, Collections.emptyList());
  }

  /**
   * Chat with an assistant.
   *
   * @param request   the AssistantChatRequest
   * @param documents uploaded documents to include
   * @return stream of chat response
   */
  public Multi<String> chat(AssistantChatRequest request, Collection<Document> documents) {

    AssistantEntity assistant;
    if (!StringUtil.isNullOrEmpty(request.getAssistantName())) {
      assistant = assistantRepository.findByName(request.getAssistantName());
    } else if (!StringUtil.isNullOrEmpty(request.getAssistantId())) {
      assistant = AssistantEntity.findById(request.getAssistantId());
    } else {
      throw new RuntimeException("Assistant Name or ID Required");
    }

    LlmConnectionEntity llmConnection = LlmConnectionEntity.findById(assistant.getLlmConnectionId());

    RetrieverConnectionEntity retrieverConnection = RetrieverConnectionEntity
                                      .findById(assistant.getRetrieverConnectionId());

    ChatBotRequest chatBotRequest = new ChatBotRequest();
    chatBotRequest.setMessage(request.getMessage());
    chatBotRequest.setContext(request.getContext());
    chatBotRequest.setRetrieverRequest(mapperUtil.toRequest(retrieverConnection));
    chatBotRequest.setModelRequest(mapperUtil.toRequest(llmConnection));
    chatBotRequest.setSystemMessage(assistant.getUserPrompt());

    return chat(chatBotRequest, documents);
  }

  /**
   * Chat with a model.
   * @param request the ChatBotRequest
   * @return stream of chat response
   */
  public Multi<String> chat(ChatBotRequest request) {
    return chat(request, Collections.emptyList());
  }

  /**
   * Chat with a model.
   * @param request the ChatBotRequest
   * @param documents documents to include
   * @return stream of chat response
   */
  public Multi<String> chat(ChatBotRequest request, Collection<Document> documents) {

    String traceId = Span.current().getSpanContext().getTraceId();
    Log.info("ChatBotService.chat for message: " + request.getMessage() + " traceId: " + traceId);
    validateRequest(request);

    StreamingChatLanguageModel llm = modelTemplateFactory.getServingRuntime(
                                                        request.getModelRequest().getServingRuntimeType())
                                                        .getChatModel(request.getModelRequest());

    // TODO: Make this configurable
    Class<? extends BaseAiService> aiServiceClass = aiServicesFactory
                        .getAiService(request.getModelRequest().getModelType());

    BaseAiService aiService = prepareAiService(
        aiServiceClass,
        llm,
        Collections.singleton(request.getRetrieverRequest()),
        documents
    );

    
    try {
        String systemMessage = request.getSystemMessage() == null ? defaultSystemMessage : request.getSystemMessage();
        
        Multi<String> multi = Multi.createFrom().emitter(em -> {
            try {
                // PASO 1: Consultar modelo BERT
                em.emit("🔍 Consultando modelo BERT para determinación inicial...\n\n");
                
                BertPredictionResponse bertPrediction = consultBertModel(request.getMessage());
                
                em.emit("📊 **Predicción BERT:** " + bertPrediction.getCulpability() + 
                       " (Confianza: " + String.format("%.2f", bertPrediction.getConfidence()) + ")\n\n");
                
                // PASO 2: Construir prompt enriquecido para LLM
                String enrichedPrompt = buildEnrichedPrompt(request.getMessage(), bertPrediction);
                
                em.emit("⚖️ Validando con LLM y leyes de tránsito argentinas...\n\n");
                
                // PASO 3: Consultar LLM con prompt enriquecido
                aiService.chatToken(request.getContext(), enrichedPrompt, systemMessage)
                    .onPartialResponse(em::emit)
                    .onRetrieved(sources -> {
                        try {
                            em.emit("START_SOURCES_STRING\n");
                            em.emit(objectMapper.writeValueAsString(new ContentResponse(sources)));
                            em.emit("\nEND_SOURCES_STRING\n");
                        } catch (JsonProcessingException e) {
                            Log.error("Sources not processable: %e", e);
                        }
                    })
                    .onError(em::fail)
                    .onCompleteResponse(response -> {
                        em.complete();
                    })
                    .start();
                    
            } catch (Exception e) {
                Log.error("Error consulting BERT model", e);
                em.fail(e);
            }
        });
        
        return multi;
         
    
    } catch (Exception e) {
      Log.error("Error in ChatBotService.chat", e);
      return Multi.createFrom().failure(e);
    }
  }

  // TODO: Support non-streaming chat?

  /**
   * Prepare an AI service with zero or more content retrievers. Content retrievers may be from a
   * {@link RetrieverRequest} or may be generated from provided {@link Document} that will be split, embedded,
   * and stored in an {@link EmbeddingStore} for retrieval by the AI
   *
   * @param <T> the type of AI service to build
   * @param aiServiceClass the class of AI service to build
   * @param llm the language model to use in the AI service
   * @param retrieverRequests an optional collection of  {@link RetrieverRequest} to provide additional knowledge
   *     to the AI
   * @param documents an optional collection of {@link Document} in which the AI may infer knowledge
   * @return the built AI service
   */
  private <T extends BaseAiService> T prepareAiService(
      Class<T> aiServiceClass,
      StreamingChatLanguageModel llm,
      Collection<RetrieverRequest> retrieverRequests,
      Collection<Document> documents
  ) {
    AiServices<T> builder = AiServices.builder(aiServiceClass)
        .streamingChatLanguageModel(llm);

    List<ContentRetriever> retrievers = new ArrayList<>();

    if (retrieverRequests != null) {
      retrieverRequests.stream()
          .filter(Objects::nonNull)
          .map(ragService::getContentRetriever)
          .forEach(retrievers::add);
    }

    if (documents != null && !documents.isEmpty()) {
      // TODO: Make these configurable in the Assistant or LLM descriptions; for now, they're null to use defaults that
      //  are configured in application.properties (or simialr config sources, such as environment variables)
      Integer maxResults = null;
      Double minScore = null;

      retrievers.add(ragService.contentRetrieverForDocuments(
          documents,
          maxResults,
          minScore
      ));
    }

    if (!retrievers.isEmpty()) {
      builder.retrievalAugmentor(DefaultRetrievalAugmentor.builder()
          .queryRouter(new DefaultQueryRouter(retrievers))
          .build());
    }

    return builder.build();
  }

  private void validateRequest(ChatBotRequest request) {
    if (request.getMessage() == null) {
      throw new RuntimeException("Request Message Required");
    }
  }

  /**
   * Consulta el modelo BERT para obtener predicción inicial
   */
  private BertPredictionResponse consultBertModel(String siniestroDescription) {
      try {
          // Llamada al API REST del modelo BERT
          String bertApiUrl = configProperties.getBertApiUrl(); // ej: "http://bert-service:8080/predict"
          
          BertRequest bertRequest = new BertRequest(siniestroDescription);
          
          return restClient.post()
                  .uri(bertApiUrl)
                  .bodyValue(bertRequest)
                  .retrieve()
                  .bodyToMono(BertPredictionResponse.class)
                  .block();
                  
      } catch (Exception e) {
          Log.error("Error calling BERT model API", e);
          // Fallback en caso de error
          return new BertPredictionResponse("Indeterminado", 0.0);
      }
  }

  /**
   * Construye el prompt enriquecido con la predicción BERT
   */
  private String buildEnrichedPrompt(String originalMessage, BertPredictionResponse bertPrediction) {
      return String.format("""
          Como experto en seguros y leyes de tránsito argentinas, necesito que analices el siguiente siniestro:
          
          **DESCRIPCIÓN DEL SINIESTRO:**
          %s
          
          **PREDICCIÓN INICIAL DEL MODELO BERT:**
          - Culpabilidad: %s
          - Nivel de confianza: %.2f
          
          **INSTRUCCIONES:**
          1. Analiza la descripción del siniestro considerando las leyes de tránsito argentinas
          2. Valida si la predicción BERT es correcta o si difiere tu análisis
          3. Proporciona una justificación detallada citando los artículos específicos de la ley de tránsito
          4. Indica si CONFIRMAS, MODIFICAS o RECHAZAS la predicción del modelo BERT
          
          **FORMATO DE RESPUESTA:**
          - **Decisión Final:** [Culpable/No Culpable/Indeterminado]
          - **Validación BERT:** [Confirmada/Modificada/Rechazada]
          - **Justificación Legal:** [Cita los artículos específicos]
          - **Análisis:** [Explicación detallada del razonamiento]
          
          Responde de manera profesional y precisa, basándote únicamente en la legislación argentina de tránsito.
          """, 
          originalMessage, 
          bertPrediction.getCulpability(), 
          bertPrediction.getConfidence()
      );
  }

  // Clases de apoyo para la integración con BERT
  public static class BertRequest {
      private String description;
      
      public BertRequest(String description) {
          this.description = description;
      }
      
      // getters/setters
      public String getDescription() { return description; }
      public void setDescription(String description) { this.description = description; }
  }

  public static class BertPredictionResponse {
      private String culpability; // "Culpable", "No Culpable", "Indeterminado"
      private double confidence;   // 0.0 - 1.0
      
      public BertPredictionResponse() {}
      
      public BertPredictionResponse(String culpability, double confidence) {
          this.culpability = culpability;
          this.confidence = confidence;
      }
      
      // getters/setters
      public String getCulpability() { return culpability; }
      public void setCulpability(String culpability) { this.culpability = culpability; }
      
      public double getConfidence() { return confidence; }
      public void setConfidence(double confidence) { this.confidence = confidence; }
  } 
  
}
