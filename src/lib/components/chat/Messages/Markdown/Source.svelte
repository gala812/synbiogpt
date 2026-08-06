<script lang="ts">
	export let token;
	export let onClick: Function = () => {};

	let id = '';
	let displayId = '';

	function extractDataAttribute(input) {
		// Use a regular expression to extract the value of the `data` attribute
		const match = input.match(/data="([^"]*)"/);
		// Check if a match was found and return the first captured group
		return match ? match[1] : null;
	}

	$: id = extractDataAttribute(token.text);

	function resolveDisplayId(rawId: string) {
		const normalized = `${rawId ?? ''}`.trim();
		if (!normalized) return '';
		if (/^\d+$/.test(normalized)) return normalized;

		const sourceEl = document.getElementById(`source-${normalized}`);
		const indexed = sourceEl?.getAttribute('data-citation-index');
		if (indexed && /^\d+$/.test(indexed)) {
			return indexed;
		}

		return normalized;
	}

	$: displayId = resolveDisplayId(id);
</script>

{#if displayId}
	<button
		class="inline-flex align-super text-[10px] leading-none font-semibold mx-0.5 px-1 py-0.5 rounded-md dark:bg-white/10 dark:text-white/70 dark:hover:text-white bg-gray-100 text-black/70 hover:text-black hover:bg-gray-200 dark:hover:bg-white/20 transition cursor-pointer"
		on:click={(event) => {
			event.stopPropagation();
			onClick(id, event);
		}}
	>
		<span>
			[{displayId}]
		</span>
	</button>
{/if}
