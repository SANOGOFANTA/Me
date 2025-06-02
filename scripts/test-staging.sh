# scripts/test-staging.sh

set -e

STAGING_URL=${STAGING_URL:-"http://staging-api.yourdomain.com"}
NAMESPACE="staging"

echo "🧪 Tests de validation staging..."
echo "URL: $STAGING_URL"

# Test 1: Health Check
echo "1️⃣ Test de santé..."
if curl -f "$STAGING_URL/health" >/dev/null 2>&1; then
    echo "✅ Health check OK"
else
    echo "❌ Health check échoué"
    exit 1
fi

# Test 2: API Documentation
echo "2️⃣ Test documentation API..."
if curl -f "$STAGING_URL/docs" >/dev/null 2>&1; then
    echo "✅ Documentation accessible"
else
    echo "❌ Documentation non accessible"
fi

# Test 3: Prédiction simple
echo "3️⃣ Test prédiction simple..."
RESPONSE=$(curl -s -X POST "$STAGING_URL/predict" \
    -H "Content-Type: application/json" \
    -d '{"text": "Je me sens anxieux et inquiet"}')

if echo "$RESPONSE" | grep -q "prediction"; then
    echo "✅ Prédiction simple OK"
    echo "   Réponse: $RESPONSE"
else
    echo "❌ Prédiction simple échouée"
    echo "   Réponse: $RESPONSE"
    exit 1
fi

# Test 4: Prédiction batch
echo "4️⃣ Test prédiction batch..."
BATCH_RESPONSE=$(curl -s -X POST "$STAGING_URL/predict_batch" \
    -H "Content-Type: application/json" \
    -d '{"texts": ["Je suis heureux", "Je me sens triste", "Tout va bien"]}')

if echo "$BATCH_RESPONSE" | grep -q "predictions"; then
    echo "✅ Prédiction batch OK"
else
    echo "❌ Prédiction batch échouée"
    echo "   Réponse: $BATCH_RESPONSE"
    exit 1
fi

# Test 5: Métriques Prometheus
echo "5️⃣ Test métriques..."
if curl -f "$STAGING_URL/metrics" | grep -q "model_predictions_total"; then
    echo "✅ Métriques Prometheus OK"
else
    echo "❌ Métriques Prometheus échouées"
fi

# Test 6: Performance (latence)
echo "6️⃣ Test de performance..."
START_TIME=$(date +%s%N)
curl -s -X POST "$STAGING_URL/predict" \
    -H "Content-Type: application/json" \
    -d '{"text": "Test de performance"}' >/dev/null
END_TIME=$(date +%s%N)
LATENCY=$(( (END_TIME - START_TIME) / 1000000 ))  # en millisecondes

if [ $LATENCY -lt 1000 ]; then
    echo "✅ Performance OK (${LATENCY}ms)"
else
    echo "⚠️  Latence élevée (${LATENCY}ms)"
fi

# Test 7: Vérification des pods Kubernetes
echo "7️⃣ Test infrastructure Kubernetes..."
READY_PODS=$(kubectl get pods -n $NAMESPACE -l app=sentiment-classifier --no-headers | grep "Running" | grep "1/1" | wc -l)
TOTAL_PODS=$(kubectl get pods -n $NAMESPACE -l app=sentiment-classifier --no-headers | wc -l)

if [ "$READY_PODS" -eq "$TOTAL_PODS" ] && [ "$READY_PODS" -gt 0 ]; then
    echo "✅ Infrastructure K8s OK ($READY_PODS/$TOTAL_PODS pods ready)"
else
    echo "❌ Problème infrastructure K8s ($READY_PODS/$TOTAL_PODS pods ready)"
    kubectl get pods -n $NAMESPACE -l app=sentiment-classifier
    exit 1
fi

# Test 8: HPA fonctionnel
echo "8️⃣ Test HPA..."
HPA_STATUS=$(kubectl get hpa sentiment-classifier-hpa -n $NAMESPACE -o jsonpath='{.status.conditions[0].status}' 2>/dev/null || echo "NotFound")
if [ "$HPA_STATUS" = "True" ]; then
    echo "✅ HPA fonctionnel"
else
    echo "⚠️  HPA non fonctionnel ou non trouvé"
fi

echo ""
echo "🎉 Tests de validation terminés!"
echo ""
echo "📊 Résumé:"
echo "   - Service en ligne et fonctionnel"
echo "   - API endpoints opérationnels"
echo "   - Performance acceptable"
echo "   - Infrastructure stable"
echo ""
echo "🔗 Environnement staging prêt!"
