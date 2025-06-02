#!/bin/bash
# scripts/deploy-staging.sh

set -e

# Configuration
NAMESPACE="staging"
IMAGE_TAG=${IMAGE_TAG:-"staging"}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-"your-registry.com"}
KUBECTL_CONTEXT=${KUBECTL_CONTEXT:-"staging-cluster"}

echo "🚀 Déploiement en staging..."
echo "Namespace: $NAMESPACE"
echo "Image Tag: $IMAGE_TAG"
echo "Registry: $DOCKER_REGISTRY"

# Vérifier les prérequis
command -v kubectl >/dev/null 2>&1 || { echo "kubectl requis mais non installé"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "docker requis mais non installé"; exit 1; }

# Se connecter au bon cluster
kubectl config use-context $KUBECTL_CONTEXT

# Créer le namespace s'il n'existe pas
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Appliquer les configurations de base
echo "📋 Application des configurations..."
kubectl apply -f k8s/staging/namespace.yaml
kubectl apply -f k8s/staging/configmap.yaml
kubectl apply -f k8s/staging/secrets.yaml
kubectl apply -f k8s/staging/serviceaccount.yaml
kubectl apply -f k8s/staging/rbac.yaml
kubectl apply -f k8s/staging/persistentvolume.yaml
kubectl apply -f k8s/staging/networkpolicy.yaml

# Attendre que les PVC soient disponibles
echo "⏳ Attente des volumes persistants..."
kubectl wait --for=condition=Bound pvc/model-pvc -n $NAMESPACE --timeout=120s
kubectl wait --for=condition=Bound pvc/monitoring-pvc -n $NAMESPACE --timeout=120s

# Mettre à jour l'image dans le deployment
echo "🔄 Mise à jour de l'image..."
sed "s|your-registry.com/sentiment-classifier:staging|$DOCKER_REGISTRY/sentiment-classifier:$IMAGE_TAG|g" \
    k8s/staging/deployment.yaml | kubectl apply -f -

# Appliquer le reste des configurations
kubectl apply -f k8s/staging/service.yaml
kubectl apply -f k8s/staging/ingress.yaml
kubectl apply -f k8s/staging/hpa.yaml

# Si Prometheus Operator est installé
if kubectl get crd servicemonitors.monitoring.coreos.com >/dev/null 2>&1; then
    kubectl apply -f k8s/staging/servicemonitor.yaml
    echo "✅ ServiceMonitor appliqué"
fi

# Attendre que le déploiement soit prêt
echo "⏳ Attente du déploiement..."
kubectl rollout status deployment/sentiment-classifier -n $NAMESPACE --timeout=300s

# Vérifier que les pods sont prêts
kubectl wait --for=condition=ready pod -l app=sentiment-classifier -n $NAMESPACE --timeout=300s

# Vérifier la santé du service
echo "🔍 Vérification de la santé du service..."
SERVICE_IP=$(kubectl get svc sentiment-classifier-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [ -z "$SERVICE_IP" ]; then
    SERVICE_IP=$(kubectl get svc sentiment-classifier-service -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
fi

# Port-forward pour tester localement si pas d'IP externe
kubectl port-forward svc/sentiment-classifier-service 8080:80 -n $NAMESPACE &
PF_PID=$!
sleep 5

# Test de santé
for i in {1..10}; do
    if curl -f http://localhost:8080/health >/dev/null 2>&1; then
        echo "✅ Service en bonne santé!"
        break
    else
        echo "⏳ Tentative $i/10 - Service pas encore prêt..."
        sleep 10
    fi
    
    if [ $i -eq 10 ]; then
        echo "❌ Service non accessible après 10 tentatives"
        kill $PF_PID 2>/dev/null || true
        exit 1
    fi
done

# Nettoyer le port-forward
kill $PF_PID 2>/dev/null || true

# Afficher les informations de déploiement
echo ""
echo "🎉 Déploiement staging terminé avec succès!"
echo ""
echo "📊 Informations du déploiement:"
kubectl get pods -n $NAMESPACE -l app=sentiment-classifier
echo ""
kubectl get svc -n $NAMESPACE
echo ""
kubectl get ingress -n $NAMESPACE

# Afficher les logs récents
echo ""
echo "📝 Logs récents:"
kubectl logs -n $NAMESPACE -l app=sentiment-classifier --tail=20

echo ""
echo "🔗 URLs utiles:"
echo "   API Health: http://staging-api.yourdomain.com/health"
echo "   API Docs: http://staging-api.yourdomain.com/docs"
echo "   Métriques: http://staging-api.yourdomain.com/metrics"

echo ""
echo "🛠️ Commandes utiles:"
echo "   Logs: kubectl logs -f -n $NAMESPACE -l app=sentiment-classifier"
echo "   Scale: kubectl scale deployment sentiment-classifier --replicas=3 -n $NAMESPACE"
echo "   Delete: kubectl delete -f k8s/staging/ || kubectl delete namespace $NAMESPACE"


#!/bin/bash


#!/bin/bash
