<script lang="ts">
	import { getContext } from 'svelte';
	import Selector from './Knowledge/Selector.svelte';
	import FileItem from '$lib/components/common/FileItem.svelte';

	export let selectedKnowledge = [];
	export let collections = [];

	const i18n = getContext('i18n');

	// ✅ 已屏蔽：tag 相关逻辑
	// let availableTags = [];
	// onMount(async () => { ... });
	// const toggleTag = (...) => { ... };
</script>

<div>
	<div class="flex w-full justify-between mb-1">
		<div class=" self-center text-sm font-semibold">{$i18n.t('Knowledge')}</div>
	</div>

	<div class=" text-xs dark:text-gray-500">
		{$i18n.t('To attach knowledge base here, add them to the "Knowledge" workspace first.')}
	</div>

	<div class="flex flex-col">
		{#if selectedKnowledge?.length > 0}
			<div class=" flex flex-col gap-3 mt-2">
				{#each selectedKnowledge as file, fileIdx}
					<div class="flex flex-col gap-2">
						<FileItem
							{file}
							name={file.name}
							type={file?.legacy
								? `Legacy${file.type ? ` ${file.type}` : ''}`
								: (file?.type ?? 'Collection')}
							dismissible
							on:dismiss={(e) => {
								selectedKnowledge = selectedKnowledge.filter((_, idx) => idx !== fileIdx);
							}}
						/>

						<!-- ✅ 已屏蔽：tag 选择 UI -->
						<!--
						{#if availableTags.length > 0}
							<div class="flex flex-wrap gap-1.5 text-xs">
								{#each availableTags as tag (tag.id)}
									<button
										type="button"
										class="px-2 py-1 rounded-full border transition
											{(file.tags ?? []).includes(tag.id)
												? 'bg-gray-200 dark:bg-gray-800 border-gray-300 dark:border-gray-700'
												: 'bg-transparent hover:bg-gray-100 dark:hover:bg-gray-900 border-gray-200 dark:border-gray-800'}"
										on:click={() => toggleTag(file.id, tag.id)}
										aria-pressed={(file.tags ?? []).includes(tag.id)}
									>
										{tag.name}
									</button>
								{/each}
							</div>
						{/if}
						-->
					</div>
				{/each}
			</div>
		{/if}

		<div class="flex flex-wrap text-sm font-medium gap-1.5 mt-2">
			<Selector
				on:select={(e) => {
					const item = e.detail;

					if (!selectedKnowledge.find((k) => k.id === item.id)) {
						selectedKnowledge = [
							...selectedKnowledge,
							{
								...item
							}
						];
					}
				}}
			>
				<button
					class=" px-3.5 py-1.5 font-medium hover:bg-black/5 dark:hover:bg-white/5 outline outline-1 outline-gray-100 dark:outline-gray-850 rounded-3xl"
					type="button">{$i18n.t('Select Knowledge')}</button
				>
			</Selector>
		</div>
	</div>
</div>
