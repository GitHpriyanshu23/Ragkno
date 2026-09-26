import React, { createContext, useContext, useState, useEffect } from 'react'

export const TRANSLATIONS = {
  en: {
    // Sidebar
    newChat: 'New Chat',
    connectApps: 'Connect Apps',
    yourChats: 'Your chats',
    noChats: 'No chats yet',
    settings: 'Settings',
    helpFaq: 'Help & FAQ',
    learnMore: 'Learn more',
    giveFeedback: 'Give feedback',
    logOut: 'Log out',

    // Chat
    askAnything: 'Ask anything',
    disclaimer: 'RagKno synthesizes knowledge from your connected documents. Verify important info.',
    sources: 'Sources',
    allSources: 'All indexed sources',
    searchSources: 'Search indexed sources...',
    addSources: 'Add sources',
    clear: 'Clear',
    refresh: 'Refresh',
    emptyGreeting: 'What do you want to explore today?',
    suggestion1: 'Summarize key decisions from latest notes',
    suggestion2: 'Find requirements and specifications in docs',
    suggestion3: 'Extract action items from recent meetings',
    suggestion4: 'Compare architecture proposals across files',
    activeSources: 'Active Sources',
    noSourcesFound: 'No indexed sources matching filter',
    uploadLocal: 'Upload local files',
    freePlan: 'Free plan · Upgrade',

    // Settings Modal Tabs
    tabGeneral: 'General',
    tabAiRag: 'AI & RAG Engine',
    tabData: 'Data Controls',
    tabHelp: 'Help & FAQ',
    tabAbout: 'About RagKno',

    // Settings Modal Rows
    searchSettings: 'search settings',
    appearance: 'appearance',
    appearanceDesc: 'choose your color scheme',
    themeSystem: 'system',
    themeDark: 'dark',
    themeLight: 'light',

    language: 'language',
    languageDesc: 'set interface language',

    streaming: 'response streaming',
    streamingDesc: 'show tokens as they generate',

    reranker: 'semantic reranking',
    rerankerDesc: 'reorder results with neural reranker',

    topK: 'retrieval depth (top-k)',
    topKDesc: 'number of chunks sent to the model',

    model: 'reasoning model',
    modelDesc: 'gemini model used for answers',

    export: 'export conversations',
    exportDesc: 'download all chats as json',
    exportBtn: 'export json',

    syncDrive: 'sync google drive',
    syncDriveDesc: 'sync your connected drive files',
    syncBtn: 'sync now',
    syncing: 'syncing...',

    manageData: 'manage data',
    manageDataDesc: 'open document indexer and drive dashboard',
    openDataControls: 'open data controls',

    clearHistory: 'clear chat history',
    clearHistoryDesc: 'erase all conversations from this browser',
    clearBtn: 'clear all chats',

    diagnostics: 'diagnostics',
    diagnosticsDesc: 'check vector store and api status',
    runHealthCheck: 'run health check',
    testing: 'testing...',

    accountPrivacy: 'account & privacy',
    isolatedBadge: 'isolated & encrypted',
    shortcuts: 'keyboard shortcuts',
    shortcutSendMessage: 'send message',
    shortcutNewChat: 'new chat',
    shortcutToggleSidebar: 'toggle sidebar',
    shortcutSettings: 'settings',
    shortcutClose: 'close modal',

    aboutVersion: 'v2.4.0 (enterprise rag)',
    aboutEngine: 'powered by gemini 2.5 flash + chromadb + google drive api.',
    aboutDocs: 'view docs',
  },
  es: {
    // Sidebar
    newChat: 'Nuevo chat',
    connectApps: 'Conectar apps',
    yourChats: 'Tus conversaciones',
    noChats: 'Sin conversaciones aún',
    settings: 'Configuración',
    helpFaq: 'Ayuda y preguntas frecuentes',
    learnMore: 'Más información',
    giveFeedback: 'Enviar comentarios',
    logOut: 'Cerrar sesión',

    // Chat
    askAnything: 'Pregunta cualquier cosa',
    disclaimer: 'RagKno sintetiza conocimiento de tu Google Drive y documentos conectados. Verifica datos importantes.',
    sources: 'Fuentes',
    allSources: 'Todas las fuentes indexadas',
    searchSources: 'Buscar fuentes indexadas...',
    addSources: 'Agregar fuentes',
    clear: 'Limpiar',
    refresh: 'Actualizar',
    emptyGreeting: '¿Qué te gustaría explorar hoy?',
    suggestion1: 'Resumir decisiones clave de las últimas notas',
    suggestion2: 'Buscar requisitos y especificaciones en documentos',
    suggestion3: 'Extraer tareas pendientes de reuniones recientes',
    suggestion4: 'Comparar propuestas de arquitectura entre archivos',
    activeSources: 'Fuentes activas',
    noSourcesFound: 'No se encontraron fuentes indexadas',
    uploadLocal: 'Subir archivos locales',
    freePlan: 'Plan gratuito · Mejorar',

    // Settings Modal Tabs
    tabGeneral: 'General',
    tabAiRag: 'Motor de IA y RAG',
    tabData: 'Control de datos',
    tabHelp: 'Ayuda y FAQ',
    tabAbout: 'Acerca de RagKno',

    // Settings Modal Rows
    searchSettings: 'Buscar en configuración',
    appearance: 'Apariencia',
    appearanceDesc: 'Personaliza la combinación de colores de la interfaz',
    themeSystem: 'Sistema',
    themeDark: 'Oscuro',
    themeLight: 'Claro',

    language: 'Idioma',
    languageDesc: 'Idioma de la interfaz para menús, etiquetas y ajustes',

    streaming: 'Activar streaming de respuestas',
    streamingDesc: 'Muestra los tokens de IA en tiempo real mientras se generan',

    reranker: 'Reordenamiento semántico CrossEncoder',
    rerankerDesc: 'Reordena fragmentos vectoriales con un reclasificador neuronal de alta precisión',

    topK: 'Presupuesto de contexto (Top-K)',
    topKDesc: 'Número de fragmentos de documentos inyectados en la ventana de razonamiento',

    model: 'Modelo de razonamiento IA',
    modelDesc: 'Modelo Gemini subyacente que impulsa la síntesis RAG',

    export: 'Exportar conversaciones',
    exportDesc: 'Descarga todas tus conversaciones locales como un archivo JSON',
    exportBtn: 'Exportar JSON',

    syncDrive: 'Sincronizar Google Drive',
    syncDriveDesc: 'Dispara una sincronización diferencial en segundo plano con tus archivos',
    syncBtn: 'Sincronizar ahora',
    syncing: 'Sincronizando...',

    manageData: 'Administrar datos conectados',
    manageDataDesc: 'Abrir indexador de documentos y panel de Google Drive',
    openDataControls: 'Abrir control de datos',

    clearHistory: 'Borrar historial de chat',
    clearHistoryDesc: 'Elimina permanentemente todas las conversaciones de este navegador',
    clearBtn: 'Borrar todos los chats',

    diagnostics: 'Diagnóstico del sistema',
    diagnosticsDesc: 'Verificar el estado del almacén de vectores ChromaDB y la API',
    runHealthCheck: 'Ejecutar comprobación',
    testing: 'Comprobando...',

    accountPrivacy: 'Protección de cuenta y privacidad',
    isolatedBadge: 'Aislado y cifrado',
    shortcuts: 'Atajos de teclado',
    shortcutSendMessage: 'Enviar mensaje',
    shortcutNewChat: 'Nuevo chat',
    shortcutToggleSidebar: 'Alternar barra lateral',
    shortcutSettings: 'Panel de configuración',
    shortcutClose: 'Cerrar ventana emergente',

    aboutVersion: 'Versión 2.4.0 (Enterprise RAG)',
    aboutEngine: 'Desarrollado con Gemini 2.5 Flash + búsqueda vectorial ChromaDB + Google Drive API.',
    aboutDocs: 'Ver documentación del proyecto',
  },
  fr: {
    // Sidebar
    newChat: 'Nouvelle discussion',
    connectApps: 'Connecter des applis',
    yourChats: 'Vos discussions',
    noChats: 'Aucune discussion',
    settings: 'Paramètres',
    helpFaq: 'Aide et FAQ',
    learnMore: 'En savoir plus',
    giveFeedback: 'Donner un avis',
    logOut: 'Déconnexion',

    // Chat
    askAnything: 'Posez n’importe quelle question',
    disclaimer: 'RagKno synthétise les connaissances de votre Google Drive et documents connectés. Vérifiez les informations importantes.',
    sources: 'Sources',
    allSources: 'Toutes les sources indexées',
    searchSources: 'Rechercher des sources indexées...',
    addSources: 'Ajouter des sources',
    clear: 'Effacer',
    refresh: 'Actualiser',
    emptyGreeting: 'Que souhaitez-vous explorer aujourd’hui ?',
    suggestion1: 'Résumer les décisions clés des dernières notes',
    suggestion2: 'Trouver les exigences et spécifications dans les documents',
    suggestion3: 'Extraire les points d’action des réunions récentes',
    suggestion4: 'Comparer les propositions d’architecture entre les fichiers',
    activeSources: 'Sources actives',
    noSourcesFound: 'Aucune source trouvée',
    uploadLocal: 'Téléverser des fichiers locaux',
    freePlan: 'Formule gratuite · Mettre à niveau',

    // Settings Modal Tabs
    tabGeneral: 'Général',
    tabAiRag: 'Moteur IA et RAG',
    tabData: 'Gestion des données',
    tabHelp: 'Aide et FAQ',
    tabAbout: 'À propos de RagKno',

    // Settings Modal Rows
    searchSettings: 'Rechercher dans les paramètres',
    appearance: 'Apparence',
    appearanceDesc: 'Personnaliser le thème de couleur de l’interface',
    themeSystem: 'Système',
    themeDark: 'Sombre',
    themeLight: 'Clair',

    language: 'Langue',
    languageDesc: 'Langue de l’interface pour les menus, libellés et paramètres',

    streaming: 'Activer le streaming des réponses',
    streamingDesc: 'Afficher les jetons IA en temps réel lors de leur génération',

    reranker: 'Reclassement sémantique CrossEncoder',
    rerankerDesc: 'Reclasse les fragments de vecteurs avec un modèle neuronal de précision',

    topK: 'Budget de contexte (Top-K)',
    topKDesc: 'Nombre de fragments de documents injectés dans la fenêtre de raisonnement',

    model: 'Modèle de raisonnement IA',
    modelDesc: 'Modèle Gemini sous-jacent alimentant la synthèse RAG',

    export: 'Exporter les discussions',
    exportDesc: 'Télécharger toutes vos discussions locales sous forme d’archive JSON',
    exportBtn: 'Exporter JSON',

    syncDrive: 'Synchroniser Google Drive',
    syncDriveDesc: 'Déclencher une synchronisation d’arrière-plan avec vos fichiers Google Drive',
    syncBtn: 'Synchroniser maintenant',
    syncing: 'Synchronisation...',

    manageData: 'Gérer les données connectées',
    manageDataDesc: 'Ouvrir l’indexeur de documents et le tableau de bord Google Drive',
    openDataControls: 'Ouvrir la gestion des données',

    clearHistory: 'Effacer l’historique des discussions',
    clearHistoryDesc: 'Supprimer définitivement toutes les discussions de ce navigateur',
    clearBtn: 'Effacer toutes les discussions',

    diagnostics: 'Diagnostics du système',
    diagnosticsDesc: 'Vérifier l’état de ChromaDB et la connexion API',
    runHealthCheck: 'Exécuter le test de santé',
    testing: 'Test en cours...',

    accountPrivacy: 'Protection du compte et de la vie privée',
    isolatedBadge: 'Isolé et chiffré',
    shortcuts: 'Raccourcis clavier',
    shortcutSendMessage: 'Envoyer le message',
    shortcutNewChat: 'Nouvelle discussion',
    shortcutToggleSidebar: 'Afficher/masquer la barre latérale',
    shortcutSettings: 'Fenêtre des paramètres',
    shortcutClose: 'Fermer la boîte de dialogue',

    aboutVersion: 'Version 2.4.0 (Enterprise RAG)',
    aboutEngine: 'Propulsé par Gemini 2.5 Flash + recherche vectorielle ChromaDB + Google Drive API.',
    aboutDocs: 'Consulter la documentation du projet',
  },
  de: {
    // Sidebar
    newChat: 'Neuer Chat',
    connectApps: 'Apps verbinden',
    yourChats: 'Ihre Chats',
    noChats: 'Noch keine Chats',
    settings: 'Einstellungen',
    helpFaq: 'Hilfe & FAQ',
    learnMore: 'Mehr erfahren',
    giveFeedback: 'Feedback geben',
    logOut: 'Abmelden',

    // Chat
    askAnything: 'Fragen Sie alles',
    disclaimer: 'RagKno fasst Wissen aus Google Drive und Dokumenten zusammen. Überprüfen Sie wichtige Infos.',
    sources: 'Quellen',
    allSources: 'Alle indexierten Quellen',
    searchSources: 'Indexierte Quellen durchsuchen...',
    addSources: 'Quellen hinzufügen',
    clear: 'Löschen',
    refresh: 'Aktualisieren',
    emptyGreeting: 'Was möchten Sie heute entdecken?',
    suggestion1: 'Wichtige Entscheidungen aus den neuesten Notizen zusammenfassen',
    suggestion2: 'Anforderungen und Spezifikationen in Dokumenten finden',
    suggestion3: 'Aktionspunkte aus den letzten Meetings extrahieren',
    suggestion4: 'Architekturvorschläge dateiübergreifend vergleichen',
    activeSources: 'Aktive Quellen',
    noSourcesFound: 'Keine passenden indexierten Quellen gefunden',
    uploadLocal: 'Lokale Dateien hochladen',
    freePlan: 'Kostenloser Plan · Upgrade',

    // Settings Modal Tabs
    tabGeneral: 'Allgemein',
    tabAiRag: 'KI- & RAG-Engine',
    tabData: 'Datenkontrollen',
    tabHelp: 'Hilfe & FAQ',
    tabAbout: 'Über RagKno',

    // Settings Modal Rows
    searchSettings: 'Einstellungen durchsuchen',
    appearance: 'Erscheinungsbild',
    appearanceDesc: 'Farbschema der Benutzeroberfläche anpassen',
    themeSystem: 'System',
    themeDark: 'Dunkel',
    themeLight: 'Hell',

    language: 'Sprache',
    languageDesc: 'Sprache der Benutzeroberfläche für Menüs, Beschriftungen und Einstellungen',

    streaming: 'Antwort-Streaming aktivieren',
    streamingDesc: 'KI-Tokens in Echtzeit anzeigen, während sie generiert werden',

    reranker: 'Semantisches CrossEncoder-Reranking',
    rerankerDesc: 'Ordnet abgerufene Vektoren mit hochpräzisem neuronalem Reranker neu',

    topK: 'Kontextabruf-Budget (Top-K)',
    topKDesc: 'Anzahl semantischer Dokumentabschnitte im Kontextfenster',

    model: 'KI-Denkmodell',
    modelDesc: 'Zugrundeliegendes Gemini-Modell für RAG-Synthese',

    export: 'Unterhaltungen exportieren',
    exportDesc: 'Alle lokalen Chat-Verläufe als JSON-Archiv herunterladen',
    exportBtn: 'JSON exportieren',

    syncDrive: 'Google Drive synchronisieren',
    syncDriveDesc: 'Sofortige Hintergrund-Synchronisierung mit Google Drive auslösen',
    syncBtn: 'Jetzt synchronisieren',
    syncing: 'Synchronisiere...',

    manageData: 'Verbundene Daten verwalten',
    manageDataDesc: 'Dokumentindexer und Google Drive Dashboard öffnen',
    openDataControls: 'Datenkontrollen öffnen',

    clearHistory: 'Chat-Verlauf löschen',
    clearHistoryDesc: 'Alle Chats und Daten dauerhaft aus diesem Browser löschen',
    clearBtn: 'Alle Chats löschen',

    diagnostics: 'Systemdiagnose',
    diagnosticsDesc: 'ChromaDB-Vektorspeicher und API-Status prüfen',
    runHealthCheck: 'Systemprüfung ausführen',
    testing: 'Prüfe...',

    accountPrivacy: 'Konto- & Datenschutz',
    isolatedBadge: 'Isoliert & verschlüsselt',
    shortcuts: 'Tastenkombinationen',
    shortcutSendMessage: 'Nachricht senden',
    shortcutNewChat: 'Neuer Chat',
    shortcutToggleSidebar: 'Seitenleiste umschalten',
    shortcutSettings: 'Einstellungsdialog',
    shortcutClose: 'Modal / Menü schließen',

    aboutVersion: 'Version 2.4.0 (Enterprise RAG)',
    aboutEngine: 'Unterstützt von Gemini 2.5 Flash + ChromaDB Vektorsuche + Google Drive API.',
    aboutDocs: 'Projektdokumentation ansehen',
  },
  hi: {
    // Sidebar
    newChat: 'नई बातचीत',
    connectApps: 'ऐप्स कनेक्ट करें',
    yourChats: 'आपकी बातचीत',
    noChats: 'अभी कोई बातचीत नहीं',
    settings: 'सेटिंग्स',
    helpFaq: 'सहायता और अक्सर पूछे जाने वाले प्रश्न',
    learnMore: 'अधिक जानें',
    giveFeedback: 'प्रतिक्रिया दें',
    logOut: 'लॉग आउट',

    // Chat
    askAnything: 'कुछ भी पूछें',
    disclaimer: 'RagKno आपके कनेक्टेड गूगल ड्राइव और दस्तावेज़ों से जानकारी संश्लेषित करता है। महत्वपूर्ण जानकारी सत्यापित करें।',
    sources: 'स्रोत',
    allSources: 'सभी इंडेक्स किए गए स्रोत',
    searchSources: 'इंडेक्स किए गए स्रोत खोजें...',
    addSources: 'स्रोत जोड़ें',
    clear: 'साफ़ करें',
    refresh: 'रीफ़्रेश करें',
    emptyGreeting: 'आज आप क्या खोजना चाहते हैं?',
    suggestion1: 'नवीनतम नोट्स से मुख्य निर्णयों का सारांश दें',
    suggestion2: 'दस्तावेज़ों में आवश्यकताएं और विनिर्देश खोजें',
    suggestion3: 'हाल की बैठकों से कार्य मदों को निकालें',
    suggestion4: 'फ़ाइलों में वास्तुकला प्रस्तावों की तुलना करें',
    activeSources: 'सक्रिय स्रोत',
    noSourcesFound: 'कोई इंडेक्स किया गया स्रोत नहीं मिला',
    uploadLocal: 'स्थानीय फ़ाइलें अपलोड करें',
    freePlan: 'निःशुल्क प्लान · अपग्रेड करें',

    // Settings Modal Tabs
    tabGeneral: 'सामान्य',
    tabAiRag: 'एआई और आरएजी इंजन',
    tabData: 'डेटा नियंत्रण',
    tabHelp: 'सहायता और अक्सर पूछे जाने वाले प्रश्न',
    tabAbout: 'RagKno के बारे में',

    // Settings Modal Rows
    searchSettings: 'सेटिंग्स खोजें',
    appearance: 'दिखावट',
    appearanceDesc: 'इंटरफ़ेस रंग योजना अनुकूलित करें',
    themeSystem: 'सिस्टम',
    themeDark: 'डार्क',
    themeLight: 'लाइट',

    language: 'भाषा',
    languageDesc: 'मेनू, लेबल और सेटिंग्स के लिए इंटरफ़ेस भाषा',

    streaming: 'रिस्पॉन्स स्ट्रीमिंग सक्षम करें',
    streamingDesc: 'उत्पन्न होते ही वास्तविक समय में एआई टोकन प्रदर्शित करें',

    reranker: 'सिमेंटिक क्रॉसएन्कोडर रीरैंकिंग',
    rerankerDesc: 'उच्च-सटीक तंत्रिका रीरैंकर के साथ पुनर्प्राप्त वेक्टर विखंडों को व्यवस्थित करता है',

    topK: 'संदर्भ पुनर्प्राप्ति बजट (Top-K)',
    topKDesc: 'तर्क खिड़की में इंजेक्ट किए गए दस्तावेज़ विखंडों की संख्या',

    model: 'एआई तर्क मॉडल',
    modelDesc: 'मल्टी-हॉप आरएजी संश्लेषण को शक्ति प्रदान करने वाला जेमिनी मॉडल',

    export: 'बातचीत निर्यात करें',
    exportDesc: 'अपने सभी स्थानीय चैट थ्रेड को JSON फ़ाइल के रूप में डाउनलोड करें',
    exportBtn: 'JSON निर्यात करें',

    syncDrive: 'गूगल ड्राइव सिंक करें',
    syncDriveDesc: 'अपने Google Drive फ़ाइलों के साथ पृष्ठभूमि में तुरंत सिंक ट्रिगर करें',
    syncBtn: 'अभी सिंक करें',
    syncing: 'सिंक हो रहा है...',

    manageData: 'कनेक्टेड डेटा प्रबंधित करें',
    manageDataDesc: 'दस्तावेज़ अनुक्रमणिका और गूगल ड्राइव सिंक डैशबोर्ड खोलें',
    openDataControls: 'डेटा नियंत्रण खोलें',

    clearHistory: 'चैट इतिहास साफ़ करें',
    clearHistoryDesc: 'इस ब्राउज़र से सभी चैट बातचीत और मेमोरी स्थायी रूप से मिटाएं',
    clearBtn: 'सभी चैट साफ़ करें',

    diagnostics: 'सिस्टम निदान',
    diagnosticsDesc: 'ChromaDB वेक्टर स्टोर और एपीआई कनेक्शन स्थिति सत्यापित करें',
    runHealthCheck: 'हेल्थ चेक चलाएं',
    testing: 'जांच जारी है...',

    accountPrivacy: 'खाता और गोपनीयता सुरक्षा',
    isolatedBadge: 'पृथक और एन्क्रिप्टेड',
    shortcuts: 'कीबोर्ड शॉर्टकट',
    shortcutSendMessage: 'संदेश भेजें',
    shortcutNewChat: 'नई बातचीत',
    shortcutToggleSidebar: 'साइडबार टॉगल करें',
    shortcutSettings: 'सेटिंग्स विंडो',
    shortcutClose: 'विंडो बंद करें',

    aboutVersion: 'संस्करण 2.4.0 (Enterprise RAG)',
    aboutEngine: 'Gemini 2.5 Flash + ChromaDB वेक्टर खोज + Google Drive API द्वारा संचालित।',
    aboutDocs: 'परियोजना दस्तावेज़ देखें',
  },
}

export function detectLanguage(preference) {
  if (preference && preference !== 'auto' && TRANSLATIONS[preference]) {
    return preference
  }
  if (typeof navigator !== 'undefined') {
    const navLang = (navigator.language || navigator.userLanguage || '').toLowerCase()
    if (navLang.startsWith('es')) return 'es'
    if (navLang.startsWith('fr')) return 'fr'
    if (navLang.startsWith('de')) return 'de'
    if (navLang.startsWith('hi')) return 'hi'
  }
  return 'en'
}

const I18nContext = createContext({
  language: 'auto',
  effectiveLang: 'en',
  setLanguage: () => {},
  t: (key) => key,
})

export function I18nProvider({ children }) {
  const [language, setLanguageState] = useState(() => {
    try {
      return localStorage.getItem('ragkno_lang') || 'auto'
    } catch {
      return 'auto'
    }
  })

  const effectiveLang = detectLanguage(language)

  const setLanguage = (newLang) => {
    setLanguageState(newLang)
    try {
      localStorage.setItem('ragkno_lang', newLang)
    } catch {}
    window.dispatchEvent(new CustomEvent('ragkno-lang-change', { detail: newLang }))
  }

  useEffect(() => {
    const handleStorage = (e) => {
      if (e.key === 'ragkno_lang' && e.newValue) {
        setLanguageState(e.newValue)
      }
    }
    const handleCustom = (e) => {
      if (e.detail) {
        setLanguageState(e.detail)
      }
    }
    window.addEventListener('storage', handleStorage)
    window.addEventListener('ragkno-lang-change', handleCustom)
    return () => {
      window.removeEventListener('storage', handleStorage)
      window.removeEventListener('ragkno-lang-change', handleCustom)
    }
  }, [])

  const t = (key) => {
    const currentDict = TRANSLATIONS[effectiveLang] || TRANSLATIONS.en
    return currentDict[key] || TRANSLATIONS.en[key] || key
  }

  return (
    <I18nContext.Provider value={{ language, effectiveLang, setLanguage, t }}>
      {children}
    </I18nContext.Provider>
  )
}

export function useI18n() {
  return useContext(I18nContext)
}
