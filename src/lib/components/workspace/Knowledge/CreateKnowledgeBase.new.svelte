<script>
	import { goto } from '$app/navigation';
	import { getContext } from 'svelte';
	const i18n = getContext('i18n');

	import { WEBUI_BASE_URL } from '$lib/constants';
	import { getModels } from '$lib/apis';
	import { createNewKnowledge, getKnowledgeBases } from '$lib/apis/knowledge';
	import { generateOpenAIChatCompletion } from '$lib/apis/openai';
	import { toast } from 'svelte-sonner';
	import { config, knowledge, models, settings, user } from '$lib/stores';
	import AccessControl from '../common/AccessControl.svelte';

	let loading = false;
	let name = '';
	let description = '';
	let accessControl = null;
	let tagSource = false;

	let generatedKeywords = [];
	let tagSeed = '';
	let generatingKeywords = false;

	const resolveSuggestionModelId = (modelList = $models) => {
		const normalizedToRawId = new Map();
		for (const model of modelList ?? []) {
			const rawId = typeof model?.id === 'string' ? model.id : '';
			const normalized = rawId.trim();
			if (normalized && !normalizedToRawId.has(normalized)) {
				normalizedToRawId.set(normalized, rawId);
			}
		}
		if (normalizedToRawId.size === 0) {
			return '';
		}

		const settingCandidates = Array.isArray($settings?.models) ? $settings.models : [];
		const configCandidates = String($config?.default_models ?? '')
			.split(',')
			.map((id) => id.trim())
			.filter(Boolean);
		const modelCandidates = (modelList ?? [])
			.map((model) => (typeof model?.id === 'string' ? model.id : ''))
			.filter(Boolean);
		const candidates = [...settingCandidates, ...configCandidates, ...modelCandidates];

		for (const candidate of candidates) {
			const normalized = typeof candidate === 'string' ? candidate.trim() : '';
			if (normalized && normalizedToRawId.has(normalized)) {
				return normalizedToRawId.get(normalized) || '';
			}
		}
		return '';
	};

	const extractKeywords = (content) => {
		if (!content) return [];

		let text = content.trim();
		const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
		if (fenced?.[1]) {
			text = fenced[1].trim();
		}

		try {
			const parsed = JSON.parse(text);
			if (Array.isArray(parsed)) {
				return [...new Set(parsed.map((it) => String(it).trim()).filter(Boolean))].slice(0, 8);
			}
			if (parsed && Array.isArray(parsed.tags)) {
				return [...new Set(parsed.tags.map((it) => String(it).trim()).filter(Boolean))].slice(0, 8);
			}
		} catch (_) {
			// Ignore parse failure and fallback.
		}

		return [
			...new Set(
				text
					.split(/[\n,;|]/g)
					.map((it) => it.replace(/^[-*\d.\s]+/, '').replace(/^["'`]+|["'`]+$/g, '').trim())
					.filter(Boolean)
			)
		].slice(0, 8);
	};

	const generateKeywordsFromSeed = async () => {
		if (!tagSeed.trim()) {
			toast.error($i18n.t('Please enter one sentence first.'));
			return;
		}

		// Always refresh model list at request time to avoid stale IDs from local cache/settings.
		const latestModels = await getModels(localStorage.token).catch(() => []);
		if (Array.isArray(latestModels) && latestModels.length > 0) {
			models.set(latestModels);
		}

		let modelId = resolveSuggestionModelId(
			Array.isArray(latestModels) && latestModels.length > 0 ? latestModels : $models
		);
		if (!modelId) {
			toast.error($i18n.t('No available model for keyword generation.'));
			return;
		}

		generatingKeywords = true;
		try {
			let [res] = await generateOpenAIChatCompletion(
				localStorage.token,
				{
					model: modelId,
					stream: false,
					messages: [
						{
							role: 'system',
							content:
								'You generate concise knowledge-base tags. Return only a JSON array of 4 to 8 short keywords. No explanation.'
						},
						{
							role: 'user',
							content: `Knowledge base name: ${name || '(empty)'}\nSeed sentence: ${tagSeed}\nOutput: JSON array only.`
						}
					]
				},
				`${WEBUI_BASE_URL}/api`
			);

			if (!res || !res.ok) {
				let detail = '';
				try {
					const err = await res.json();
					detail = err?.detail || err?.error?.message || '';
				} catch (_) {
					// Ignore parse failure and use fallback detail.
				}

				// Retry once with a fresh model list when selected model is stale.
				if ((detail || '').toLowerCase().includes('model not found')) {
					const latestModels = await getModels(localStorage.token).catch(() => []);
					if (Array.isArray(latestModels) && latestModels.length > 0) {
						models.set(latestModels);
					}
					const retryModelId = resolveSuggestionModelId(
						Array.isArray(latestModels) && latestModels.length > 0 ? latestModels : $models
					);
					if (retryModelId) {
						modelId = retryModelId;
						[res] = await generateOpenAIChatCompletion(
							localStorage.token,
							{
								model: modelId,
								stream: false,
								messages: [
									{
										role: 'system',
										content:
											'You generate concise knowledge-base tags. Return only a JSON array of 4 to 8 short keywords. No explanation.'
									},
									{
										role: 'user',
										content: `Knowledge base name: ${name || '(empty)'}\nSeed sentence: ${tagSeed}\nOutput: JSON array only.`
									}
								]
							},
							`${WEBUI_BASE_URL}/api`
						);
					}
				}

				if (!res || !res.ok) {
					throw new Error(detail ? `${detail} | model=${modelId}` : 'keyword_generation_failed');
				}
			}

			const data = await res.json();
			const content = data?.choices?.[0]?.message?.content ?? '';
			const keywords = extractKeywords(content);
			if (keywords.length === 0) {
				throw new Error('empty_keywords');
			}

			generatedKeywords = keywords;
			description = keywords.join(', ');
			toast.success($i18n.t('Keywords generated and filled into description.'));
		} catch (e) {
			console.error(e);
			const detail = typeof e?.message === 'string' ? e.message.trim() : '';
			const msg = detail && detail !== 'keyword_generation_failed' ? detail : null;
			toast.error(
				msg
					? `${$i18n.t('Failed to generate keywords.')} (${msg})`
					: $i18n.t('Failed to generate keywords.')
			);
		} finally {
			generatingKeywords = false;
		}
	};

	const submitHandler = async () => {
		loading = true;

		if (name.trim() === '' || description.trim() === '') {
			toast.error($i18n.t('Please fill in all fields.'));
			name = '';
			description = '';
			loading = false;
			return;
		}

		const res = await createNewKnowledge(
			localStorage.token,
			name,
			description,
			accessControl,
			[],
			tagSource
		).catch((e) => {
			toast.error(e);
		});

		if (res) {
			toast.success($i18n.t('Knowledge created successfully.'));
			knowledge.set(await getKnowledgeBases(localStorage.token));
			goto(`/workspace/knowledge/${res.id}`);
		}

		loading = false;
	};
</script>

<div class="w-full max-h-full">
	<button
		class="flex space-x-1"
		on:click={() => {
			goto('/workspace/knowledge');
		}}
	>
		<div class="self-center">
			<svg
				xmlns="http://www.w3.org/2000/svg"
				viewBox="0 0 20 20"
				fill="currentColor"
				class="w-4 h-4"
			>
				<path
					fill-rule="evenodd"
					d="M17 10a.75.75 0 01-.75.75H5.612l4.158 3.96a.75.75 0 11-1.04 1.08l-5.5-5.25a.75.75 0 010-1.08l5.5-5.25a.75.75 0 111.04 1.08L5.612 9.25H16.25A.75.75 0 0117 10z"
					clip-rule="evenodd"
				/>
			</svg>
		</div>
		<div class="self-center font-medium text-sm">{$i18n.t('Back')}</div>
	</button>

	<form
		class="flex flex-col max-w-lg mx-auto mt-10 mb-10"
		on:submit|preventDefault={() => {
			submitHandler();
		}}
	>
		<div class="w-full flex flex-col justify-center">
			<div class="text-2xl font-medium font-primary mb-2.5">{$i18n.t('Create a knowledge base')}</div>

			<div class="w-full flex flex-col gap-2.5">
				<div class="w-full">
					<div class="text-sm mb-2">{$i18n.t('What are you working on?')}</div>

					<div class="w-full mt-1">
						<input
							class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-none"
							type="text"
							bind:value={name}
							placeholder={$i18n.t('Name your knowledge base')}
							required
						/>
					</div>
				</div>

				<div>
					<div class="text-sm mb-2">{$i18n.t('What are you trying to achieve?')}</div>

					<div class="w-full mt-1">
						<textarea
							class="w-full resize-none rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-none"
							rows="4"
							bind:value={description}
							placeholder={$i18n.t('Describe your knowledge base and objectives')}
							required
						/>
					</div>
				</div>
			</div>
		</div>

		<div class="mt-3">
			<div class="text-sm mb-2">{$i18n.t('Generate tags with one sentence')}</div>
			<div class="text-xs text-gray-500 dark:text-gray-400">
				{$i18n.t('Enter one sentence. AI will generate tags and fill the description box.')}
			</div>
			<div class="mt-2 flex gap-2">
				<input
					class="w-full rounded-lg py-2 px-4 text-sm bg-gray-50 dark:text-gray-300 dark:bg-gray-850 outline-none"
					type="text"
					bind:value={tagSeed}
					placeholder={$i18n.t('Example: build a chemistry synthesis paper knowledge base')}
				/>
				<button
					type="button"
					class="text-sm px-3 py-2 rounded-lg bg-gray-50 hover:bg-gray-100 dark:bg-gray-850 dark:hover:bg-gray-800 disabled:opacity-60 disabled:cursor-not-allowed"
					on:click={generateKeywordsFromSeed}
					disabled={generatingKeywords}
				>
					{generatingKeywords ? $i18n.t('Generating...') : $i18n.t('Generate')}
				</button>
			</div>

			{#if generatedKeywords.length > 0}
				<div class="mt-2 flex flex-wrap gap-2">
					{#each generatedKeywords as keyword (keyword)}
						<span
							class="text-xs px-2.5 py-1.5 rounded-full border border-gray-200 dark:border-gray-800 bg-gray-100 dark:bg-gray-900"
						>
							{keyword}
						</span>
					{/each}
				</div>
			{/if}
		</div>

		{#if $user?.role === 'admin'}
			<div class="mt-3">
				<label class="flex items-center gap-2 text-sm">
					<input
						type="checkbox"
						class="rounded border-gray-300 dark:border-gray-700"
						bind:checked={tagSource}
					/>
					<span>{$i18n.t('Used as a basic knowledge base')}</span>
				</label>
				<!-- <div class="text-xs text-gray-500 dark:text-gray-400 mt-1">
					{$i18n.t('Tags will be collected from this knowledge base for filtering and discovery.')}
				</div> -->
			</div>
		{/if}

		<div class="mt-2">
			<div class="px-3 py-2 bg-gray-50 dark:bg-gray-950 rounded-lg">
				<AccessControl bind:accessControl />
			</div>
		</div>

		<div class="flex justify-end mt-2">
			<div>
				<button
					class="text-sm px-4 py-2 transition rounded-lg {loading
						? ' cursor-not-allowed bg-gray-100 dark:bg-gray-800'
						: ' bg-gray-50 hover:bg-gray-100 dark:bg-gray-850 dark:hover:bg-gray-800'} flex"
					type="submit"
					disabled={loading}
				>
					<div class="self-center font-medium">{$i18n.t('Create Knowledge')}</div>

					{#if loading}
						<div class="ml-1.5 self-center">
							<svg
								class="w-4 h-4"
								viewBox="0 0 24 24"
								fill="currentColor"
								xmlns="http://www.w3.org/2000/svg"
								><style>
									.spinner_ajPY {
										transform-origin: center;
										animation: spinner_AtaB 0.75s infinite linear;
									}
									@keyframes spinner_AtaB {
										100% {
											transform: rotate(360deg);
										}
									}
								</style><path
									d="M12,1A11,11,0,1,0,23,12,11,11,0,0,0,12,1Zm0,19a8,8,0,1,1,8-8A8,8,0,0,1,12,20Z"
									opacity=".25"
								/><path
									d="M10.14,1.16a11,11,0,0,0-9,8.92A1.59,1.59,0,0,0,2.46,12,1.52,1.52,0,0,0,4.11,10.7a8,8,0,0,1,6.66-6.61A1.42,1.42,0,0,0,12,2.69h0A1.57,1.57,0,0,0,10.14,1.16Z"
									class="spinner_ajPY"
								/></svg
							>
						</div>
					{/if}
				</button>
			</div>
		</div>
	</form>
</div>
