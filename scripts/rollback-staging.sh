# scripts/rollback-staging.sh

set -e

NAMESPACE="staging"
REVISION=${1:-"previous"}

echo "🔄 Rollback du déploiement staging..."
echo "Namespace: $NAMESPACE"
echo "Révision: $REVISION"

# Vérifier l'historique des déploiements
echo "📋 Historique des déploiements:"
kubectl rollout history deployment/sentiment-classifier -n $NAMESPACE

# Effectuer le rollback
if [ "$REVISION" = "previous" ]; then
    echo "⏪ Rollback vers la révision précédente..."
    kubectl rollout undo deployment/sentiment-classifier -n $NAMESPACE
else
    echo "⏪ Rollback vers la révision $REVISION..."
    kubectl rollout undo deployment/sentiment-classifier -n $NAMESPACE --to-revision=$REVISION
fi

# Attendre que le rollback soit terminé
echo "⏳ Attente du rollback..."
kubectl rollout status deployment/sentiment-classifier -n $NAMESPACE --timeout=300s

# Vérifier la santé après rollback
echo "🔍 Vérification post-rollback..."
kubectl wait --for=condition=ready pod -l app=sentiment-classifier -n $NAMESPACE --timeout=300s

# Test de santé rapide
echo "🧪 Test de santé..."
kubectl port-forward svc/sentiment-classifier-service 8080:80 -n $NAMESPACE &
PF_PID=$!
sleep 5

if curl -f http://localhost:8080/health >/dev/null 2>&1; then
    echo "✅ Rollback réussi - Service opérationnel"
else
    echo "❌ Rollback échoué - Service non accessible"
    kill $PF_PID 2>/dev/null || true
    exit 1
fi

kill $PF_PID 2>/dev/null || true

echo ""
echo "🎉 Rollback terminé avec succès!"
kubectl get pods -n $NAMESPACE -l app=sentiment-classifier