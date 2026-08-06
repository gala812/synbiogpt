<script>
	import { io } from 'socket.io-client';
	import { spring } from 'svelte/motion';

	let loadingProgress = spring(0, {
		stiffness: 0.05
	});

	import { onMount, tick, setContext } from 'svelte';
	import {
		config,
		user,
		theme,
		WEBUI_NAME,
		mobile,
		socket,
		activeUserCount,
		USAGE_POOL
	} from '$lib/stores';
	import { goto } from '$app/navigation';
	import { page } from '$app/stores';
	import { Toaster, toast } from 'svelte-sonner';
  import FirstLogin from '$lib/components/layout/Overlay/FirstLogin.svelte';
	import { getBackendConfig } from '$lib/apis';
	import { getSessionUser } from '$lib/apis/auths';

	import '../tailwind.css';
	import '../app.css';

	import 'tippy.js/dist/tippy.css';

	import { WEBUI_BASE_URL, WEBUI_HOSTNAME } from '$lib/constants';
	import i18n, { initI18n, getLanguages } from '$lib/i18n';
	import { bestMatchingLanguage } from '$lib/utils';

	setContext('i18n', i18n);

	let loaded = false;
	const BREAKPOINT = 768;

	const setupSocket = () => {
		const _socket = io(`${WEBUI_BASE_URL}` || undefined, {
			reconnection: true,
			reconnectionDelay: 1000,
			reconnectionDelayMax: 5000,
			randomizationFactor: 0.5,
			path: '/ws/socket.io',
			auth: { token: localStorage.token }
		});

		socket.set(_socket);

		_socket.on('connect_error', (err) => {
			console.log('connect_error', err);
		});

		_socket.on('connect', () => {
			console.log('connected', _socket.id);
		});

		_socket.on('reconnect_attempt', (attempt) => {
			console.log('reconnect_attempt', attempt);
		});

		_socket.on('reconnect_failed', () => {
			console.log('reconnect_failed');
		});

		_socket.on('disconnect', (reason, details) => {
			console.log(`Socket ${_socket.id} disconnected due to ${reason}`);
			if (details) {
				console.log('Additional details:', details);
			}
		});

		_socket.on('user-count', (data) => {
			console.log('user-count', data);
			activeUserCount.set(data.count);
		});

		_socket.on('usage', (data) => {
			console.log('usage', data);
			USAGE_POOL.set(data['models']);
		});
	};

  let showFirstLoginOverlay = false;

  $: if ($user) {
    const isHomePage = $page.url.pathname === '/';
    const isFirstLogin = !!$user && !$user.oauth_sub;
    const isPending = $user.role === 'pending';

    showFirstLoginOverlay = isHomePage && isFirstLogin && !isPending;
  }


	onMount(async () => {
		theme.set(localStorage.theme);

		mobile.set(window.innerWidth < BREAKPOINT);
		const onResize = () => {
			if (window.innerWidth < BREAKPOINT) {
				mobile.set(true);
			} else {
				mobile.set(false);
			}
		};

		window.addEventListener('resize', onResize);

		// Initialize i18n even if config retrieval fails,
		// so `/error` can still render translated text.
		initI18n();

		let sessionUser = null;
		if (localStorage.token) {
			sessionUser = await getSessionUser(localStorage.token).catch((error) => {
				toast.error(error);
				return null;
			});

			console.log('sessionUser from API:', sessionUser);

			if (sessionUser) {
				await user.set(sessionUser);
			} else {
				localStorage.removeItem('token');
				await goto('/auth');
			}
		}

		const [backendConfig, languages] = await Promise.all([
			getBackendConfig().catch((error) => {
				console.error('Error loading backend config:', error);
				return null;
			}),
			!localStorage.locale ? getLanguages().catch(() => []) : Promise.resolve([])
		]);

		if (!localStorage.locale) {
			const browserLanguages = navigator.languages
				? navigator.languages
				: [navigator.language || navigator.userLanguage];
			const lang = backendConfig?.default_locale
				? backendConfig.default_locale
				: bestMatchingLanguage(languages, browserLanguages, 'en-US');
			$i18n.changeLanguage(lang);
		}

		if (backendConfig) {
			await config.set(backendConfig);
			await WEBUI_NAME.set(backendConfig.name);

			if ($config) {
				setupSocket();

				if (!localStorage.token) {
					// Don't redirect if we're already on the auth page
					// Needed because we pass in tokens from OAuth logins via URL fragments
					if ($page.url.pathname !== '/auth') {
						await goto('/auth');
					}
				}
			}
		} else {
			// Redirect to /error when Backend Not Detected
			await goto(`/error`);
		}

		await tick();

		if (
			document.documentElement.classList.contains('her') &&
			document.getElementById('progress-bar')
		) {
			loadingProgress.subscribe((value) => {
				const progressBar = document.getElementById('progress-bar');

				if (progressBar) {
					progressBar.style.width = `${value}%`;
				}
			});

			await loadingProgress.set(100);

			document.getElementById('splash-screen')?.remove();

			const audio = new Audio(`/audio/greeting.mp3`);
			const playAudio = () => {
				audio.play();
				document.removeEventListener('click', playAudio);
			};

			document.addEventListener('click', playAudio);

			loaded = true;
		} else {
			document.getElementById('splash-screen')?.remove();
			loaded = true;
		}

		return () => {
			window.removeEventListener('resize', onResize);
		};
	});
</script>

<svelte:head>
	<title>welcome</title>
	<link crossorigin="anonymous" rel="icon" href="{WEBUI_BASE_URL}/static/default.png" />

	<!-- rosepine themes have been disabled as it's not up to date with our latest version. -->
	<!-- feel free to make a PR to fix if anyone wants to see it return -->
	<!-- <link rel="stylesheet" type="text/css" href="/themes/rosepine.css" />
	<link rel="stylesheet" type="text/css" href="/themes/rosepine-dawn.css" /> -->
</svelte:head>

{#if loaded}
	<slot />
{/if}

{#if loaded && showFirstLoginOverlay}
	<FirstLogin onClose={() => (showFirstLoginOverlay = false)} />
{/if}
<Toaster
	theme={$theme.includes('dark')
		? 'dark'
		: $theme === 'system'
			? window.matchMedia('(prefers-color-scheme: dark)').matches
				? 'dark'
				: 'light'
			: 'light'}
	richColors
	position="top-center"
/>
